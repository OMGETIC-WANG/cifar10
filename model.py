import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import typing as T


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: T.Sequence[int],
        activation: T.Callable[[jax.Array], jax.Array],
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List()
        self.activation = activation

        features = [in_features] + list(hidden_features) + [out_features]
        for din, dout in zip(features[:-1], features[1:]):
            self.layers.append(nnx.Linear(din, dout, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class ResLinear(nnx.Module):
    def __init__(
        self,
        features: int,
        activation: T.Callable[[jax.Array], jax.Array],
        *,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.activation = activation

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nnx.BatchNorm(features, rngs=rngs)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jax.lax.cond(
            self.use_batchnorm,
            lambda inst, x: inst.batchnorm(x),
            lambda inst, x: x,
            self,
            self.linear(x),
        )
        y = self.activation(y)
        y = jax.lax.cond(
            self.use_dropout, lambda inst, x: inst.dropout(x), lambda inst, x: x, self, y
        )
        return x + y


class AttachShortcut(nnx.Module):
    def __init__(self, module: nnx.Module):
        self.module = module

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.module(x)


class TransformerBlock(nnx.Module):
    def __init__(self, in_features: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            in_features,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            decode=False,
        )
        self.fnn = MLP(in_features, in_features, [in_features * 4], nnx.gelu, rngs=rngs)

        self.pre_attention_norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.pre_fnn_norm = nnx.LayerNorm(in_features, rngs=rngs)

        self.fnn_dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = x + self.attention(self.pre_attention_norm(x))
        y = self.fnn(self.pre_fnn_norm(x))
        return x + self.fnn_dropout(y)

    def CalcClsToken(self, x: jax.Array, cls_token_index: int = -1):
        batch_size = x.shape[0]
        cls_tokens = x[:, cls_token_index, :]
        pre_attention_normed = self.pre_attention_norm(x)
        cls_tokens = cls_tokens + self.attention(
            pre_attention_normed[:, cls_token_index, :][:, None, :],
            pre_attention_normed,
            pre_attention_normed,
        ).reshape(batch_size, -1)
        return cls_tokens + self.fnn_dropout(self.fnn(self.pre_fnn_norm(cls_tokens)))


class PreCNN(nnx.Module):
    def __init__(self, model_features: int, rngs: nnx.Rngs):
        self.cnn = nnx.Sequential(
            nnx.Conv(3, model_features, (9, 9), padding="VALID", rngs=rngs),
            nnx.leaky_relu,
            nnx.LayerNorm(model_features, rngs=rngs),
        )
        _, width, height, channels = nnx.eval_shape(
            lambda m, x: m(x), self.cnn, jnp.zeros((1, 32, 32, 3))
        ).shape
        self.weight_pool = nnx.Param(
            jax.random.uniform(rngs.params(), ((width // 2) * (height // 2),))
        )
        self.norm = nnx.LayerNorm(channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.cnn(x)
        batch_size, _, _, channels = x.shape
        x = x.reshape(batch_size, -1, 4, channels)
        x = jnp.einsum("bsnc,s->bsc", x, self.weight_pool) / 4
        x = x.reshape(batch_size, -1, channels)
        x = self.norm(x)
        return x


class CIFAR10Model(nnx.Module):
    def __init__(
        self,
        model_features: int,
        num_heads: int,
        num_encoder: int,
        input_shape,
        rngs: nnx.Rngs,
    ):
        self.cnn = nnx.Sequential(
            PreCNN(model_features, rngs),
            nnx.Dropout(0.4, rngs=rngs),
        )

        _, seqlen, _ = nnx.eval_shape(lambda m, x: m(x), self.cnn, jnp.zeros(input_shape)).shape
        # self.pos_embedding = nnx.Param(
        #     nnx.initializers.normal(stddev=0.02)(rngs.params(), (seqlen, model_features))
        # )
        self.pos_weight = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (seqlen, model_features))
        )
        self.pos_bias = nnx.Param(jnp.zeros((seqlen, model_features)))

        self.cls_token = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, model_features))
        )

        self.encoders = nnx.List([
            TransformerBlock(model_features, num_heads, 0.4, rngs) for _ in range(num_encoder)
        ])

        mlp_dropout_rate = 0.4
        self.target_logits_mlp = nnx.Sequential(
            nnx.LayerNorm(model_features, rngs=rngs),
            nnx.Linear(model_features, model_features * 4, rngs=rngs),
            nnx.gelu,
            nnx.LayerNorm(model_features * 4, rngs=rngs),
            nnx.Dropout(mlp_dropout_rate, rngs=rngs),
            nnx.Linear(model_features * 4, model_features * 4, rngs=rngs),
            nnx.gelu,
            nnx.LayerNorm(model_features * 4, rngs=rngs),
            nnx.Linear(model_features * 4, 10, rngs=rngs),
        )

        self.model_features = model_features

    def __call__(self, x: jax.Array):
        x = self.cnn(x)
        batch_size, input_seq_len, _ = x.shape
        # x += self.pos_embedding[:input_seq_len][None, ...]
        x = x + self.pos_weight * x + self.pos_bias

        x = jnp.concatenate(
            [jnp.full((batch_size, 1, self.model_features), self.cls_token[...]), x], axis=1
        )

        for encoder in self.encoders[:-1]:
            x = encoder(x)
        x = self.encoders[-1].CalcClsToken(x)

        return self.target_logits_mlp(x)

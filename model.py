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
    def __init__(
        self,
        *fns: T.Callable[..., T.Any],
        activation: T.Callable[[jax.Array], jax.Array] = lambda x: x,
        norm_t: T.Optional[T.Type] = nnx.LayerNorm,
        shortcut_norm_t: T.Optional[T.Type] = nnx.LayerNorm,
        in_out: tuple[int, int],
        rngs: T.Optional[nnx.Rngs] = None,
    ):
        in_channels, out_channels = in_out
        self.module = nnx.Sequential(*fns)

        self.norm = norm_t(out_channels, rngs=rngs) if norm_t is not None else None
        self.shortcut_norm = (
            shortcut_norm_t(out_channels, rngs=rngs) if shortcut_norm_t is not None else None
        )

        self.use_channel_proj = in_channels != out_channels
        if self.use_channel_proj:
            assert in_channels > 0 and out_channels > 0
            assert rngs is not None
            self.channel_proj = nnx.Linear(in_channels, out_channels, use_bias=False, rngs=rngs)

        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)
        if self.use_channel_proj:
            x = self.channel_proj(x)
        if self.shortcut_norm is not None:
            x = self.shortcut_norm(x)
        return self.activation(x + y)


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


class MultiKernelConv(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_sizes: T.Sequence[int | T.Sequence[int]],
        *args,
        rngs: nnx.Rngs,
        use_shortcut: bool = False,
        **kwargs,
    ):
        assert out_features % len(kernel_sizes) == 0

        self.convs = nnx.List([
            nnx.Conv(
                in_features,
                out_features // len(kernel_sizes),
                kernel_size,
                padding="SAME",
                rngs=rngs,
                *args,
                **kwargs,
            )
            for kernel_size in kernel_sizes
        ])
        self.use_shortcut = use_shortcut
        if use_shortcut:
            assert in_features == out_features

    def __call__(self, x: jax.Array):
        y = jnp.concatenate([conv(x) for conv in self.convs], axis=-1)
        if self.use_shortcut:
            y += x
        return y


class PreCNN(nnx.Module):
    def __init__(self, model_features: int, rngs: nnx.Rngs):
        self.cnn = nnx.Sequential(
            MultiKernelConv(3, model_features // 2, [(3, 3), (5, 5)], rngs=rngs),
            nnx.Dropout(0.2, rngs=rngs, broadcast_dims=[1, 2]),
            AttachShortcut(
                MultiKernelConv(model_features // 2, model_features, [(3, 3), (5, 5)], rngs=rngs),
                in_out=(model_features // 2, model_features),
                activation=nnx.leaky_relu,
                rngs=rngs,
            ),
            lambda x: nnx.avg_pool(x, (2, 2), (2, 2)),
            nnx.Dropout(0.1, rngs=rngs, broadcast_dims=[1, 2]),
            AttachShortcut(
                nnx.Conv(model_features, model_features, (3, 3), rngs=rngs),
                in_out=(model_features, model_features),
                activation=nnx.leaky_relu,
                rngs=rngs,
            ),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class CIFAR10Model(nnx.Module):
    def __init__(
        self,
        model_features: int,
        num_heads: int,
        num_encoder: int,
        input_shape,
        *,
        rngs: nnx.Rngs,
        cnn_dropout_rate: float = 0.2,
        encoder_dropout_rate: float = 0.2,
        pre_mlp_dropout_rate: float = 0.2,
    ):
        self.cnn = nnx.Sequential(
            PreCNN(model_features, rngs),
            nnx.Dropout(cnn_dropout_rate, rngs=rngs, broadcast_dims=[1, 2]),
        )

        _, seqlen, _ = nnx.eval_shape(lambda m, x: m(x), self.cnn, jnp.zeros(input_shape)).shape
        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (seqlen, model_features))
        )

        self.encoders = nnx.List([
            TransformerBlock(model_features, num_heads, encoder_dropout_rate, rngs=rngs)
            for _ in range(num_encoder)
        ])

        self.features_weights = nnx.Param(jnp.full((seqlen, model_features), 1 / model_features))
        self.features_weights_norm = nnx.LayerNorm(model_features, rngs=rngs)

        self.target_logits_mlp = nnx.Sequential(
            nnx.Dropout(pre_mlp_dropout_rate, rngs=rngs),
            MLP(model_features, 10, [model_features * 4], nnx.gelu, rngs=rngs),
        )

        self.model_features = model_features

    def __call__(self, x: jax.Array):
        x = self.cnn(x)
        batch_size, input_seq_len, _ = x.shape
        x += self.pos_embedding[:input_seq_len][None, ...]

        for encoder in self.encoders:
            x = encoder(x)

        w = jax.nn.squareplus(self.features_weights[...])
        x = jnp.sum(x * w[None, ...], axis=1) / jnp.sum(w, axis=0)[None, ...]
        x = self.features_weights_norm(x)

        return self.target_logits_mlp(x)

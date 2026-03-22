import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import typing as T
from double_connection import DoubleConnectionShortcut


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


class SEConv(nnx.Module):
    def __init__(
        self,
        conv: T.Callable[[jax.Array], jax.Array],
        scale_mlp: T.Optional[T.Callable[[jax.Array], jax.Array]] = None,
        scale_mlp_features: T.Optional[int] = None,
        rngs: T.Optional[nnx.Rngs] = None,
    ):
        self.conv = conv
        if scale_mlp is None:
            assert scale_mlp_features is not None and scale_mlp_features > 0, (
                "Default init scale_mlp requires scale_mlp_features > 0"
            )
            assert rngs is not None, "Default init scale_mlp requires rngs"

            def sigmoid_avg(x: jax.Array):
                x = nnx.sigmoid(x)
                return x / (jnp.sum(x, axis=-1, keepdims=True) + 1e-8)

            self.scale_mlp = nnx.Sequential(
                MLP(
                    scale_mlp_features,
                    scale_mlp_features,
                    [scale_mlp_features * 2],
                    nnx.leaky_relu,
                    rngs=rngs,
                ),
                sigmoid_avg,
            )
        else:
            self.scale_mlp = scale_mlp

    def __call__(self, x: jax.Array):
        y = self.conv(x)
        scale = self.scale_mlp(jnp.mean(y, axis=(1, 2)))
        return y * scale[:, None, None, :]


class CIFAR10Model(nnx.Module):
    def __init__(
        self,
        *,
        model_features: int,
        before_pool_conv_count: int = 4,
        after_pool_conv_count: int = 8,
        rngs: nnx.Rngs,
        expand_channel_droprate: float = 0.1,
        cnn_conv_droprate: float = 0.1,
    ):
        self.expand_channel = nnx.Sequential(
            nnx.Conv(3, model_features, (3, 3), rngs=rngs),
            nnx.leaky_relu,
            nnx.LayerNorm(model_features, rngs=rngs),
            nnx.Dropout(expand_channel_droprate, broadcast_dims=(1, 2), rngs=rngs),
        )

        build_cnn_layer = lambda: DoubleConnectionShortcut(
            SEConv(
                nnx.Sequential(
                    nnx.Conv(model_features, model_features, (3, 3), rngs=rngs),
                    nnx.leaky_relu,
                    nnx.LayerNorm(model_features, rngs=rngs),
                ),
                scale_mlp_features=model_features,
                rngs=rngs,
            ),
            nnx.Dropout(cnn_conv_droprate, broadcast_dims=(1, 2), rngs=rngs),
            rngs=rngs,
        )
        self.cnn = nnx.Sequential(*[build_cnn_layer() for _ in range(before_pool_conv_count)])

        self.pool = lambda x: nnx.avg_pool(x, (2, 2), (2, 2))

        self.cnn2 = nnx.Sequential(*[build_cnn_layer() for _ in range(after_pool_conv_count)])

        self.squeeze_channel = nnx.Conv(model_features, 2, (1, 1), rngs=rngs)

        self.final_mlp = MLP(2 * 16 * 16, 10, [2 * 16 * 16], nnx.leaky_relu, rngs=rngs)

    def __call__(self, x):
        x = self.expand_channel(x)

        x = (x, x)
        x = self.cnn(x)
        x = (self.pool(x[0]), self.pool(x[1]))
        x = self.cnn2(x)
        x = (x[0] + x[1]) / 2

        x = self.squeeze_channel(x)
        x = x.reshape(x.shape[0], -1)

        return self.final_mlp(x)

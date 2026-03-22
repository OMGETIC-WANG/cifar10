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
                    [scale_mlp_features // 2],
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


class SeperableConv(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: T.Sequence[int] | int,
        rngs: nnx.Rngs,
    ):
        self.depthwise = nnx.vmap(lambda r: nnx.Conv(1, 1, kernel_size, rngs=nnx.Rngs(r)))(
            jax.random.split(rngs.params(), in_features)
        )
        self.pointwise = nnx.Conv(in_features, out_features, (1, 1), rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.vmap(lambda m, x: m(x), in_axes=(0, 0))(
            self.depthwise, x.transpose(3, 0, 1, 2)[..., None]
        )  # in_features, B, H, W, 1
        x = x.reshape(x.shape[:-1])  # in_features, B, H, W
        x = x.transpose(1, 2, 3, 0)  # B, H, W, in_features
        x = self.pointwise(x)  # B, H, W, out_features
        return x


class DCConvDownsample(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(features, features * 2, (2, 2), strides=(2, 2), rngs=rngs)

    def __call__(self, x1: jax.Array, x2: jax.Array):
        return self.conv(x1), self.conv(x2)


class CIFAR10Model(nnx.Module):
    def __init__(
        self,
        *,
        num_32chan_conv: int = 4,
        num_64chan_conv: int = 4,
        num_128chan_conv: int = 4,
        num_256chan_conv: int = 4,
        rngs: nnx.Rngs,
        expand_channel_droprate: float = 0.1,
        cnn_conv_droprate: float = 0.1,
    ):
        self.expand_channel = nnx.Sequential(
            nnx.Conv(3, 32, (3, 3), rngs=rngs),
            nnx.leaky_relu,
            nnx.LayerNorm(32, rngs=rngs),
            nnx.Dropout(expand_channel_droprate, broadcast_dims=(1, 2), rngs=rngs),
        )

        build_cnn_layer = lambda features: DoubleConnectionShortcut(
            SEConv(
                nnx.Sequential(
                    SeperableConv(features, features, (3, 3), rngs=rngs),
                    nnx.leaky_relu,
                    nnx.LayerNorm(features, rngs=rngs),
                ),
                scale_mlp_features=features,
                rngs=rngs,
            ),
            nnx.Dropout(cnn_conv_droprate, broadcast_dims=(1, 2), rngs=rngs),
            rngs=rngs,
        )
        build_cnn = lambda features, count: nnx.Sequential(*[
            build_cnn_layer(features) for _ in range(count)
        ])

        dc_avg_pool = lambda x1, x2: (
            nnx.avg_pool(x1, (2, 2), (2, 2)),
            nnx.avg_pool(x2, (2, 2), (2, 2)),
        )

        self.cnn = nnx.Sequential(
            build_cnn(32, num_32chan_conv),
            DCConvDownsample(32, rngs=rngs),
            build_cnn(64, num_64chan_conv),
            DCConvDownsample(64, rngs=rngs),
            build_cnn(128, num_128chan_conv),
            DCConvDownsample(128, rngs=rngs),
            build_cnn(256, num_256chan_conv),
            dc_avg_pool,
        )

        self.final_mlp = MLP(256, 10, [256 * 4], nnx.leaky_relu, rngs=rngs)

    def __call__(self, x):
        x = self.expand_channel(x)

        x = (x, x)
        x = self.cnn(x)
        x = (x[0] + x[1]) / 2
        x = jnp.mean(x, axis=(1, 2))

        return self.final_mlp(x)

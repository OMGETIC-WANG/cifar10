import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.nnx as nnx
import typing as T
from double_connection import DoubleConnectionShortcut, InitDoubleConnectionShortcutWeights


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


class MultiKernelConv2D(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_sizes: T.Sequence[T.Sequence[int]],
        *,
        rngs: nnx.Rngs,
    ):
        assert out_features % len(kernel_sizes) == 0

        merged_kernel_sizes = {}
        for ks in kernel_sizes:
            if ks in merged_kernel_sizes:
                merged_kernel_sizes[ks] += 1
            else:
                merged_kernel_sizes[ks] = 1
        for ks in kernel_sizes:
            merged_kernel_sizes[ks] *= out_features // len(kernel_sizes)

        max_kernel_size = (
            max(kernel_sizes, key=lambda x: x[0])[0],
            max(kernel_sizes, key=lambda x: x[1])[1],
        )
        self.conv = nnx.Conv(in_features, out_features, max_kernel_size, rngs=rngs)

        mask = np.zeros((*max_kernel_size, 1, out_features))
        i = 0
        for kernel_size, count in merged_kernel_sizes.items():
            hpad = (max_kernel_size[0] - kernel_size[0]) // 2
            wpad = (max_kernel_size[1] - kernel_size[1]) // 2
            mask[hpad : hpad + kernel_size[0], wpad : wpad + kernel_size[1], :, i : i + count] = 1
            i += count
        self.mask = jnp.array(mask)

    def __call__(self, x: jax.Array) -> jax.Array:
        original_kernel = self.conv.kernel[...]
        self.conv.kernel[...] *= self.mask
        out = self.conv(x)
        self.conv.kernel[...] = original_kernel
        return out


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

            self.scale_mlp = nnx.Sequential(
                MLP(
                    scale_mlp_features,
                    scale_mlp_features,
                    [scale_mlp_features // 8],
                    nnx.leaky_relu,
                    rngs=rngs,
                ),
                nnx.sigmoid,
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
        self.depthwise = nnx.Conv(
            in_features, in_features, kernel_size, feature_group_count=in_features, rngs=rngs
        )
        self.pointwise = nnx.Conv(in_features, out_features, (1, 1), rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.depthwise(x)
        x = self.pointwise(x)
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
            MultiKernelConv2D(3, 32, [(3, 3), (5, 5)], rngs=rngs),
            nnx.leaky_relu,
            nnx.LayerNorm(32, rngs=rngs),
            nnx.Dropout(expand_channel_droprate, broadcast_dims=(1, 2), rngs=rngs),
        )

        def build_cnn_layer(conv_type: type, features: int, kernel_shape=(3, 3)):
            return DoubleConnectionShortcut(
                SEConv(
                    nnx.Sequential(
                        conv_type(features, features, kernel_shape, rngs=rngs),
                        nnx.leaky_relu,
                        nnx.LayerNorm(features, rngs=rngs),
                    ),
                    scale_mlp_features=features,
                    rngs=rngs,
                ),
                nnx.Dropout(cnn_conv_droprate, broadcast_dims=(1, 2), rngs=rngs),
                rngs=rngs,
            )

        def build_cnn(conv_type, features, count, kernel_shape=(3, 3)):
            return nnx.Sequential(*[
                build_cnn_layer(conv_type, features, kernel_shape) for _ in range(count)
            ])

        dc_avg_pool = lambda x1, x2: (
            nnx.avg_pool(x1, (2, 2), (2, 2)),
            nnx.avg_pool(x2, (2, 2), (2, 2)),
        )

        self.cnn = nnx.Sequential(
            build_cnn(MultiKernelConv2D, 32, num_32chan_conv, [(3, 3), (5, 5)]),
            DCConvDownsample(32, rngs=rngs),
            build_cnn(nnx.Conv, 64, num_64chan_conv),
            DCConvDownsample(64, rngs=rngs),
            build_cnn(SeperableConv, 128, num_128chan_conv),
            DCConvDownsample(128, rngs=rngs),
            build_cnn(SeperableConv, 256, num_256chan_conv),
            dc_avg_pool,
        )

        InitDoubleConnectionShortcutWeights(self.cnn.layers)

        self.final_mlp = MLP(256, 10, [256 * 4], nnx.leaky_relu, rngs=rngs)

    def __call__(self, x):
        x = self.expand_channel(x)

        x = self.cnn(x, x)
        x = (x[0] + x[1]) / 2
        x = jnp.mean(x, axis=(1, 2))

        return self.final_mlp(x)

import jax
import jax.numpy as jnp
from flax import nnx

import typing as T


class DataStrengthenConfig(T.NamedTuple):
    max_noise: float
    salt_noise_prob: float
    flip_prob: float
    mixup_weight: float

    max_crop_width: int
    max_crop_height: int

    max_scale_size: int


@jax.jit
def AddNoise(x: jax.Array, max_noise: float, rngs: nnx.Rngs):
    noise = jax.random.uniform(rngs.params(), x.shape, minval=-max_noise, maxval=max_noise)
    x = x + noise
    x = jnp.where(x > 1.0, 1.0, x)
    x = jnp.where(x < 0.0, 0.0, x)
    return x


@jax.jit
def AddSaltNoise(x: jax.Array, salt_prob: float, rngs: nnx.Rngs):
    salt_mask = jax.random.bernoulli(rngs.params(), salt_prob, x.shape)
    return jnp.where(salt_mask, 1.0, x)


@jax.jit
def RandomHorizenFlip(x: jax.Array, flip_prob: float, rngs: nnx.Rngs):
    flip_mask = jax.random.bernoulli(rngs.params(), flip_prob, (x.shape[0],))
    flipped_x = jnp.where(flip_mask[:, None, None, None], x[:, :, ::-1, :], x)
    return flipped_x


@jax.jit
def ScaleImageDown(image: jax.Array, target_size: int):
    height, width, channels = image.shape

    scale_factor = target_size / height

    return jax.image.scale_and_translate(
        image,
        shape=(height, width, channels),
        spatial_dims=(0, 1),
        scale=jnp.array([scale_factor, scale_factor]),
        translation=jnp.array([0.0, 0.0]),
        method=jax.image.ResizeMethod.LINEAR,
        antialias=True,
    )


@jax.jit
def ScaleImagesDown(images: jax.Array, max_size: float, rngs: nnx.Rngs):
    batch_size, height, width, channels = images.shape

    random_sizes = jax.random.randint(
        rngs.params(), (batch_size,), minval=height, maxval=jnp.array(max_size, dtype=jnp.int32) + 1
    )

    return jax.vmap(ScaleImageDown)(images, random_sizes)


@jax.jit
def ShiftImage(image: jax.Array, horizen_shift: int, vertical_shift: int) -> jax.Array:
    h, w, c = image.shape
    max_horizen_shift = w // 2
    max_vertical_shift = h // 2
    padded = jnp.pad(
        image,
        (
            (max_vertical_shift, max_vertical_shift),
            (max_horizen_shift, max_horizen_shift),
            (0, 0),
        ),
        mode="edge",
    )
    start_h = max_vertical_shift - vertical_shift
    start_w = max_horizen_shift - horizen_shift
    return jax.lax.dynamic_slice(padded, (start_h, start_w, 0), (h, w, c))


@jax.jit
def RandomShiftSingleImage(
    image: jax.Array, max_horizen_shift: int, max_vertical_shift: int, random_key: jax.Array
) -> jax.Array:
    k1, k2 = jax.random.split(random_key)
    horizen_shift = jax.random.randint(k1, (), -max_horizen_shift, max_horizen_shift + 1)
    vertical_shift = jax.random.randint(k2, (), -max_vertical_shift, max_vertical_shift + 1)
    return ShiftImage(image, horizen_shift, vertical_shift)


@jax.jit
def RandomShiftImage(
    x: jax.Array, max_horizen_shift: int, max_vertical_shift: int, rngs: nnx.Rngs
) -> jax.Array:
    batch_size = x.shape[0]
    random_keys = jax.random.split(rngs.params(), batch_size)
    return jax.vmap(RandomShiftSingleImage, in_axes=(0, None, None, 0))(
        x, max_horizen_shift, max_vertical_shift, random_keys
    )


@jax.jit(static_argnames=["strengthen_config"])
def ApplyStrengthen(x: jax.Array, strengthen_config: DataStrengthenConfig, rngs: nnx.Rngs):
    if strengthen_config.max_noise > 0:
        x = AddNoise(x, strengthen_config.max_noise, rngs)
    if strengthen_config.flip_prob > 0:
        x = RandomHorizenFlip(x, strengthen_config.flip_prob, rngs)
    if strengthen_config.max_scale_size > 1.0:
        x = ScaleImagesDown(x, strengthen_config.max_scale_size, rngs)
    if strengthen_config.max_crop_width > 0 or strengthen_config.max_crop_height > 0:
        x = RandomShiftImage(
            x, strengthen_config.max_crop_width, strengthen_config.max_crop_height, rngs
        )
    if strengthen_config.salt_noise_prob > 0:
        x = AddSaltNoise(x, strengthen_config.salt_noise_prob, rngs)
    return x


@nnx.jit(static_argnames=["strengthen_config"])
def Mixup(
    x1: jax.Array,
    y1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
    rngs: nnx.Rngs,
    strengthen_config: DataStrengthenConfig,
):
    weight = rngs.beta(
        strengthen_config.mixup_weight, strengthen_config.mixup_weight, (x1.shape[0], 1, 1, 1)
    )
    label_weight = weight.reshape(-1, 1)

    x_mixed = x1 * weight + x2 * (1 - weight)
    y_mixed = nnx.one_hot(y1, 10) * label_weight + nnx.one_hot(y2, 10) * (1 - label_weight)
    return x_mixed, y_mixed

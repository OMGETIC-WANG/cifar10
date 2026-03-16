from model import CIFAR10Model

import cifar10_loader

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import typing as T

from dashboard import Dashboard
import time_util

import os
import time
import model_serialization

from ml_collections import config_flags
from absl import app

import matplotlib.pyplot as plt


Model_t = T.TypeVar("Model_t", bound=nnx.Module)


class DataStrengthenConfig(T.NamedTuple):
    max_noise: float
    salt_noise_prob: float
    flip_prob: float
    mixup_weight: float
    transmix_weight: float

    max_crop_width: int
    max_crop_height: int

    max_scale_size: int


A = T.TypeVar("A")


def CondApply(cond: bool, true_fn: T.Callable[[A], A], x: A) -> A:
    return nnx.cond(cond, lambda x: true_fn(x), lambda x: x, x)


@jax.jit
def AddNoise(x: jax.Array, max_noise: float, rngs: nnx.Rngs):
    noise = jax.random.uniform(rngs.params(), x.shape, minval=0.0, maxval=max_noise)
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
    horizen_shift = jax.random.randint(random_key, (), -max_horizen_shift, max_horizen_shift)
    vertical_shift = jax.random.randint(random_key, (), -max_vertical_shift, max_vertical_shift)
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
def TrainBatch(
    model_optimizer: tuple[Model_t, nnx.Optimizer[Model_t]],
    x: jax.Array,
    y: jax.Array,
    random_key: jax.Array,
    strengthen_config: T.Optional[DataStrengthenConfig] = None,
):
    model, optimzier = model_optimizer

    if strengthen_config is not None:
        x = ApplyStrengthen(x, strengthen_config, nnx.Rngs(random_key))

    def loss_fn(model: Model_t):
        logits = model(x)
        if len(y.shape) == 1:
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        else:
            loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
        return loss, accuracy

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimzier.update(model, grads)
    return (model, optimzier), loss, accuracy


@nnx.jit(static_argnames=["batch_size", "strengthen_config"])
def TrainModel(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    rngs: nnx.Rngs,
    metrics: nnx.Metric,
    strengthen_config: T.Optional[DataStrengthenConfig] = None,
):
    indices = jnp.arange(x.shape[0])
    indices = jax.random.permutation(rngs.params(), indices)
    x, y = BatchDatas((x[indices], y[indices]), batch_size)

    random_keys = jax.random.split(rngs.params(), x.shape[0])

    @nnx.scan(in_axes=(nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0, 0))
    def TrainBatchScan(carry, x, y, random_key):
        return TrainBatch(carry, x, y, random_key, strengthen_config)

    _, losses, accuracies = TrainBatchScan((model, optimizer), x, y, random_keys)

    metrics.update(values=losses, accuracy=accuracies)


# @param mixer: (([x1,x2],[y1,y2]),random_key)->(mixed_x,mixed_y)
@nnx.jit(static_argnames=["batch_size", "mixer", "strengthen_config"])
def TrainModelWithMixup(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    mixer: T.Callable[
        [tuple[jax.Array, jax.Array], jax.Array, T.Optional[DataStrengthenConfig]],
        tuple[jax.Array, jax.Array],
    ],
    rngs: nnx.Rngs,
    metrics: nnx.Metric,
    strengthen_config: T.Optional[DataStrengthenConfig] = None,
):
    indices = jnp.arange(x.shape[0])
    indices = jax.random.permutation(rngs.params(), indices)
    x, y = BatchDatas((x[indices], y[indices]), batch_size)
    batch_count, batch_size, width, height, channels = x.shape

    x = x.reshape(batch_count // 2, 2, batch_size, width, height, channels)
    y = y.reshape(batch_count // 2, 2, batch_size)

    random_keys = jax.random.split(rngs.params(), x.shape[0])

    @nnx.scan(in_axes=(nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0, 0))
    def TrainBatchScan(carry, x, y, random_key):
        subkey, random_key = jax.random.split(random_key)
        x, y = mixer((x, y), subkey, strengthen_config)
        return TrainBatch(carry, x, y, random_key, strengthen_config)

    _, losses, accuracies = TrainBatchScan((model, optimizer), x, y, random_keys)

    metrics.update(values=losses, accuracy=accuracies)


@nnx.jit(static_argnames=["strengthen_config"])
def TransMix(
    xy: tuple[jax.Array, jax.Array],
    random_key: jax.Array,
    strengthen_config: T.Optional[DataStrengthenConfig] = None,
):
    # xy.x: (2, batch_size, weight, height, channels)
    # x1.shape: (batch_size, weight, height, channels)
    # xy.y: (2, batch_size)
    # y1.shape: (batch_size,)
    (x1, x2), (y1, y2) = xy
    weight = strengthen_config.transmix_weight if strengthen_config is not None else 0.2
    random_key, subkey = jax.random.split(random_key)
    weight = jax.random.beta(subkey, weight, weight)
    mask = jax.random.bernoulli(random_key, 1 - weight, x1.shape)

    x_mixed = jnp.where(mask, x1, x2)
    y_mixed = nnx.one_hot(y1, 10) * (1 - weight) + nnx.one_hot(y2, 10) * weight
    return x_mixed, y_mixed


def Train(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    epoch_count: int,
    *,
    rngs: nnx.Rngs,
    x_test: T.Optional[jax.Array] = None,
    y_test: T.Optional[jax.Array] = None,
    test_batch_size: T.Optional[int] = None,
    state_save_path: T.Optional[str] = None,
    state_save_per_epoch: T.Optional[int] = None,
    model_save_path: T.Optional[str] = None,
    use_graphic: bool = True,
    dashboard_block: bool = False,
    eval_per_epoch: int = 1,
    strengthen_config: T.Optional[DataStrengthenConfig] = None,
):
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(), accuracy=nnx.metrics.Average("accuracy")
    )

    if test_batch_size is None:
        test_batch_size = batch_size

    if state_save_per_epoch is not None and state_save_path is not None:
        if state_save_per_epoch <= 0:
            state_save_per_epoch = None
            state_save_path = None

    if use_graphic:
        dashboard = Dashboard(
            "Dashboard", {"Loss": ["loss"], "Accuracy": ["accuracy", "test_accuracy"]}
        )
    else:
        dashboard = None

    trainer = time_util.CountPerformance(TrainModelWithMixup)

    for epoch in range(epoch_count):
        _, timecost = trainer(
            model,
            optimizer,
            x,
            y,
            batch_size,
            TransMix,
            rngs,
            train_metrics,
            strengthen_config,
        )

        epoch_metrics = train_metrics.compute()
        train_metrics.reset()

        if state_save_path is not None and state_save_per_epoch is not None:
            if (epoch + 1) % state_save_per_epoch == 0:
                model_serialization.SaveTrainingState(
                    os.path.join(state_save_path),
                    model,
                    optimizer,
                )

        epoch_msg = f"loss: {epoch_metrics['loss']:.6f}, accuracy: {epoch_metrics['accuracy']:.6f}"
        loss_plot_dict = {"loss": epoch_metrics["loss"], "accuracy": epoch_metrics["accuracy"]}
        if (
            eval_per_epoch > 0
            and x_test is not None
            and y_test is not None
            and epoch % eval_per_epoch == 0
        ):
            model.eval()
            test_accuracy = TestModel(model, x_test, y_test, test_batch_size)
            model.train()
            epoch_msg += f", test_accuracy: {test_accuracy:.6f}"
            loss_plot_dict["test_accuracy"] = test_accuracy
        print(f"Epoch {epoch + 1}/{epoch_count} ({timecost:.2f}s) - {epoch_msg}")
        if dashboard is not None:
            dashboard.Update(loss_plot_dict)

    if model_save_path is not None:
        model_serialization.SaveModel(model_save_path, model)


@nnx.scan(in_axes=(None, 0, 0), out_axes=0)
@nnx.jit
def TestBatch(model: nnx.Module, x: jax.Array, y: jax.Array):
    logits = model(x)
    return jnp.sum(jnp.argmax(logits, axis=-1) == y)


@nnx.jit(static_argnames=["batch_size"])
def TestModel(model: nnx.Module, x: jax.Array, y: jax.Array, batch_size: int):
    testset_size = x.shape[0]
    x, y = BatchDatas((x, y), batch_size)
    res = TestBatch(model, x, y)
    return res.sum() / testset_size


def BatchDatas(xs: T.Sequence[jax.Array], batch_size: int):
    dataset_size = xs[0].shape[0]
    batch_count = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        print(
            f"Warning: dataset size {dataset_size} % batch size {batch_size} != 0, {dataset_size % batch_size} data will not be trained"
        )
    return [
        x[: batch_count * batch_size].reshape(batch_count, batch_size, *x.shape[1:]) for x in xs
    ]


def CountModuleParams(module: nnx.Module):
    params = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(leaf.size for leaf in leaves)


_CONFIG = config_flags.DEFINE_config_file(
    "config", "config.py", "Configuration file for training the model."
)


def EnableJaxOptimization(precision: str = "float16"):
    xla_flags = [
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
    ]

    existing_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = existing_flags + " " + " ".join(xla_flags)

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_enable_pgle", True)

    jax.config.update("jax_default_matmul_precision", "tensorfloat32")


def main(_):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    config = _CONFIG.value

    if config.enable_optimization:
        EnableJaxOptimization()

    print(f"-----Config-----\n{config}----------------")

    print("Initing model")
    rngs = nnx.Rngs(config.seed)

    print("Loading data")
    (x_train, y_train), (x_test, y_test) = cifar10_loader.LoadCIFAR10()
    trainset_size = x_train.shape[0]
    testset_size = x_test.shape[0]

    if not config.test_only:
        if config.use_training_model:
            model, optimizer = model_serialization.LoadTrainingState(
                config.train_state_path,
                lambda: CIFAR10Model(
                    config.model_features,
                    config.num_heads,
                    config.num_encoders,
                    (config.train_batch_size, 32, 32, 3),
                    nnx.Rngs(0),
                ),
                lambda model: nnx.Optimizer(
                    model, optax.adamw(config.learning_rate), wrt=nnx.Param
                ),
            )
        else:
            model = CIFAR10Model(
                config.model_features,
                config.num_heads,
                config.num_encoders,
                (config.train_batch_size, 32, 32, 3),
                rngs,
            )
            total_steps = config.epoch_count * (trainset_size // config.train_batch_size)
            optimizer_schedule = optax.warmup_cosine_decay_schedule(
                0.0,
                config.learning_rate,
                decay_steps=total_steps,
                warmup_steps=total_steps // 10,
                end_value=1e-6,
            )
            optimizer = nnx.Optimizer(
                model, optax.adamw(optimizer_schedule, weight_decay=1e-2), wrt=nnx.Param
            )

        print(f"Model param count: {CountModuleParams(model)}")
        print("Starting training")
        Train(
            model,
            optimizer,
            x_train,
            y_train,
            config.train_batch_size,
            config.epoch_count,
            rngs=rngs,
            x_test=x_test,
            y_test=y_test,
            test_batch_size=config.test_batch_size,
            state_save_path=config.train_state_path,
            state_save_per_epoch=config.state_save_per_epoch,
            model_save_path=os.path.join(
                config.model_save_dir, f"{time.time()}.{config.model_suffix}"
            ),
            strengthen_config=DataStrengthenConfig(
                max_noise=config.max_noise,
                salt_noise_prob=config.salt_noise_prob,
                flip_prob=config.flip_prob,
                mixup_weight=config.mixup_weight,
                transmix_weight=config.transmix_weight,
                max_crop_width=config.max_crop_width,
                max_crop_height=config.max_crop_height,
                max_scale_size=config.max_scale_size,
            ),
        )
    else:
        model = model_serialization.LoadNewestModel(
            config.model_save_dir,
            config.model_suffix,
            lambda: CIFAR10Model(
                config.model_features,
                config.num_heads,
                config.num_encoders,
                (config.test_batch_size, 32, 32, 3),
                rngs,
            ),
        )

    print("Start testing")
    model.eval()
    print(f"Test accuracy: {TestModel(model, x_test, y_test, config.test_batch_size) * 100:.4f}%")

    if config.use_graphic:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    app.run(main)

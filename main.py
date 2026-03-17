from model import CIFAR10Model

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import typing as T

import cifar10_loader
from data_strengthen import DataStrengthenConfig, ApplyStrengthen, Mixup

from dashboard import Dashboard
import time_util

import os
import time
import model_serialization

from ml_collections import config_flags
from absl import app

import matplotlib.pyplot as plt


Model_t = T.TypeVar("Model_t", bound=nnx.Module)


@nnx.jit
def TrainBatch(
    model_optimizer: tuple[Model_t, nnx.Optimizer[Model_t]],
    x: jax.Array,
    y: jax.Array,
):
    model, optimzier = model_optimizer

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


@nnx.jit(static_argnames=["batch_size", "strengthen_config", "mixer"])
def TrainModel(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    *,
    rngs: nnx.Rngs,
    metrics: nnx.Metric,
    strengthen_config: DataStrengthenConfig,
    mixer: T.Optional[
        T.Callable[
            [tuple[jax.Array, jax.Array], nnx.Rngs, DataStrengthenConfig],
            tuple[jax.Array, jax.Array],
        ]
    ] = None,
):

    indices = jnp.arange(x.shape[0])
    indices = jax.random.permutation(rngs.params(), indices)
    x, y = x[indices], y[indices]  # Shuffle

    x = ApplyStrengthen(x, strengthen_config, rngs)

    if mixer is not None:
        x, y = mixer(
            (
                x.reshape(2, x.shape[0] // 2, *x.shape[1:]),
                y.reshape(2, y.shape[0] // 2, *y.shape[1:]),
            ),
            rngs,
            strengthen_config,
        )

    x, y = BatchDatas((x, y), batch_size)

    _, losses, accuracies = nnx.scan(
        TrainBatch, in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0, 0)
    )((model, optimizer), x, y)

    metrics.update(values=losses, accuracy=accuracies)


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
    strengthen_config: DataStrengthenConfig,
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

    trainer = time_util.CountPerformance(TrainModel)

    for epoch in range(epoch_count):
        _, timecost = trainer(
            model,
            optimizer,
            x,
            y,
            batch_size,
            rngs=rngs,
            metrics=train_metrics,
            strengthen_config=strengthen_config,
            mixer=Mixup,
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
                    rngs=nnx.Rngs(0),
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
                rngs=rngs,
                cnn_dropout_rate=config.cnn_dropout_rate,
                encoder_dropout_rate=config.encoder_dropout_rate,
                pre_mlp_dropout_rate=config.pre_mlp_dropout_rate,
            )
            total_steps = config.epoch_count * (trainset_size // config.train_batch_size)
            optimizer_schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.init_learning_rate,
                peak_value=config.peek_learning_rate,
                end_value=config.end_learning_rate,
                warmup_steps=config.warmup_steps
                if config.warmup_steps != -1
                else total_steps // 10,
                decay_steps=total_steps,
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
                rngs=rngs,
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

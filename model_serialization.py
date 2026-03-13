import jax
import flax.nnx as nnx

import pathlib
import pickle

import typing as T

Model_t = T.TypeVar("Model_t", bound=nnx.Module)


def SaveModel(path: pathlib.Path | str, model: nnx.Module):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    _, state = nnx.split(model)

    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_flatten(state), f)


def LoadModel(path: pathlib.Path | str, model_init: T.Callable[[], Model_t]) -> Model_t:
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    def abstract_init():
        model = model_init()
        return nnx.split(model)

    graphdef, _ = nnx.eval_shape(abstract_init)

    with open(path, "rb") as f:
        flattened = pickle.load(f)
    state = jax.tree_util.tree_unflatten(flattened[1], flattened[0])

    model = nnx.merge(graphdef, state)
    return model


def LoadNewestModel(
    path: pathlib.Path | str, suffix: str, model_init: T.Callable[[], Model_t]
) -> Model_t:
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    model_files = list(path.glob(f"*{suffix}"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {path} with suffix {suffix}")

    newest_file = max(model_files, key=lambda f: f.stat().st_mtime)
    return LoadModel(newest_file, model_init)


def SaveTrainingState(path: pathlib.Path | str, model: Model_t, optimizer: nnx.Optimizer[Model_t]):
    SaveModel(path, model)

    optimizer_state_path = pathlib.Path(str(path) + ".optstate")
    with open(optimizer_state_path, "wb") as f:
        pickle.dump(optimizer.opt_state, f)


def LoadTrainingState(
    path: pathlib.Path | str,
    model_init: T.Callable[[], Model_t],
    optimizer_init: T.Callable[[Model_t], nnx.Optimizer[Model_t]],
) -> tuple[Model_t, nnx.Optimizer[Model_t]]:
    model = LoadModel(path, model_init)

    optimizer_state_path = pathlib.Path(str(path) + ".optstate")
    with open(optimizer_state_path, "rb") as f:
        opt_state = pickle.load(f)
    optimizer = optimizer_init(model)
    optimizer.opt_state = opt_state
    return model, optimizer

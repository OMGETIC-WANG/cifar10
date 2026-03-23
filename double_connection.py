import jax
import jax.numpy as jnp
from flax import nnx
import typing as T
from hyper_connection import _ParamReturner
from typing_extensions import overload


class DoubleConnectionShortcut(nnx.Module):
    def __init__(
        self,
        *layers,
        rngs: nnx.Rngs,
        pre_input_weight_gen: T.Optional[T.Callable[[jax.Array, jax.Array], jax.Array]] = None,
        post_layer_weight_gen: T.Optional[T.Callable[[jax.Array, jax.Array], jax.Array]] = None,
        residual_weight_gen: T.Optional[T.Callable[[jax.Array, jax.Array], jax.Array]] = None,
    ):
        self.num_split = 2
        self.module = nnx.Sequential(*layers)

        if pre_input_weight_gen is None:
            self.gen_pre_input_weight = _ParamReturner(jnp.array([]))
        else:
            self.gen_pre_input_weight = pre_input_weight_gen

        if post_layer_weight_gen is None:
            self.gen_post_layer_weight = _ParamReturner(jnp.array([]))
        else:
            self.gen_post_layer_weight = post_layer_weight_gen

        if residual_weight_gen is None:
            self.gen_residual_weight = _ParamReturner(jnp.array([]))
        else:
            self.gen_residual_weight = residual_weight_gen

    def __call__(self, x1: jax.Array, x2: jax.Array):
        pre_input_weight = self.gen_pre_input_weight(x1, x2)
        post_layer_weight = self.gen_post_layer_weight(x1, x2)
        residual_weight = self.gen_residual_weight(x1, x2)

        module_in = x1 * pre_input_weight[0] + x2 * pre_input_weight[1]

        module_out_single = self.module(module_in)

        m_out1 = module_out_single * post_layer_weight[0]
        m_out2 = module_out_single * post_layer_weight[1]

        res1 = x1 * residual_weight[0, 0] + x2 * residual_weight[0, 1]
        res2 = x1 * residual_weight[1, 0] + x2 * residual_weight[1, 1]

        return m_out1 + res1, m_out2 + res2


def _InitDoubleConnectionShortcutWeights_impl(
    layers: T.Iterable[DoubleConnectionShortcut | T.Any], i: int
):
    for layer in layers:
        if isinstance(layer, DoubleConnectionShortcut):
            assert isinstance(layer.gen_pre_input_weight, _ParamReturner)
            assert isinstance(layer.gen_post_layer_weight, _ParamReturner)
            assert isinstance(layer.gen_residual_weight, _ParamReturner)
            if i % 2 == 0:
                layer.gen_pre_input_weight = _ParamReturner(jnp.array([1.0, 1.0]))
                layer.gen_post_layer_weight = _ParamReturner(jnp.array([1.0, 0.0]))
                layer.gen_residual_weight = _ParamReturner(jnp.array([[1.0, 1.0], [1.0, 1.0]]))
            else:
                layer.gen_pre_input_weight = _ParamReturner(jnp.array([0.0, 1.0]))
                layer.gen_post_layer_weight = _ParamReturner(jnp.array([0.0, 1.0]))
                layer.gen_residual_weight = _ParamReturner(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
            i += 1
        else:
            if isinstance(layer, nnx.Sequential):
                i = _InitDoubleConnectionShortcutWeights_impl(layer.layers, i)
            elif isinstance(layer, T.Iterable) and not isinstance(layer, (str, bytes)):
                i = _InitDoubleConnectionShortcutWeights_impl(layer, i)
    return i


def InitDoubleConnectionShortcutWeights(layers: T.Iterable[DoubleConnectionShortcut | T.Any]):
    _InitDoubleConnectionShortcutWeights_impl(layers, 0)

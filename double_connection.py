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
            self.gen_pre_input_weight = _ParamReturner(rngs.normal((self.num_split,)))
        else:
            self.gen_pre_input_weight = pre_input_weight_gen

        if post_layer_weight_gen is None:
            self.gen_post_layer_weight = _ParamReturner(rngs.normal((self.num_split,)))
        else:
            self.gen_post_layer_weight = post_layer_weight_gen

        if residual_weight_gen is None:
            self.gen_residual_weight = _ParamReturner(rngs.normal((self.num_split, self.num_split)))
        else:
            self.gen_residual_weight = residual_weight_gen

    def _Call(self, x1: jax.Array, x2: jax.Array):
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

    @overload
    def __call__(self, inputs: T.Tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]: ...

    @overload
    def __call__(self, x1: jax.Array, x2: jax.Array) -> tuple[jax.Array, jax.Array]: ...

    def __call__(self, *args: T.Any, **kwargs: T.Any) -> T.Tuple[jax.Array, jax.Array]:
        if kwargs:
            if "inputs" in kwargs:
                x1, x2 = kwargs["inputs"]
            else:
                x1, x2 = kwargs["x1"], kwargs["x2"]
        elif len(args) == 1:
            x1, x2 = args[0]
        else:
            x1, x2 = args

        return self._Call(x1, x2)

import jax
import jax.numpy as jnp
from flax import nnx
import typing as T


class _ParamReturner(nnx.Module):
    def __init__(self, value: jax.Array):
        self.param = nnx.Param(value)

    def __call__(self, *args, **kwargs):
        return self.param[...]


def HyperConnectionInit_fn(x: jax.Array, num_split: int):
    return jnp.full((num_split, *x.shape), x)


class HyperConnectionInit(nnx.Module):
    def __init__(self, num_split: int):
        self.num_split = num_split

    def __call__(self, x: jax.Array):
        return jnp.full((self.num_split, *x.shape), x)


def HyperConnectionEnd_fn(x: jax.Array):
    return jnp.sum(x, axis=0)


class HyperConnectionEnd(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jax.Array):
        return HyperConnectionEnd_fn(x)


class HyperConnectionShortcut(nnx.Module):
    def __init__(
        self,
        *layers,
        num_split: int,
        rngs: nnx.Rngs,
        pre_input_weight_gen: T.Optional[T.Callable[[jax.Array], jax.Array]] = None,
        post_layer_weight_gen: T.Optional[T.Callable[[jax.Array], jax.Array]] = None,
        residual_weight_gen: T.Optional[T.Callable[[jax.Array], jax.Array]] = None,
    ):
        self.num_split = num_split
        self.module = nnx.Sequential(*layers)

        if pre_input_weight_gen is None:
            self.gen_pre_input_weight = _ParamReturner(rngs.normal((num_split,)))
        else:
            self.gen_pre_input_weight = pre_input_weight_gen

        if post_layer_weight_gen is None:
            self.gen_post_layer_weight = _ParamReturner(rngs.normal((num_split,)))
        else:
            self.gen_post_layer_weight = post_layer_weight_gen

        if residual_weight_gen is None:
            self.gen_residual_weight = _ParamReturner(rngs.normal((num_split, num_split)))
        else:
            self.gen_residual_weight = residual_weight_gen

    def __call__(self, x: jax.Array):
        input_shape = x.shape
        batch_count = input_shape[1]
        module_input_shape = input_shape[1:]

        x = x.reshape(self.num_split, batch_count, -1)

        pre_input_weight = self.gen_pre_input_weight(x)
        post_layer_weight = self.gen_post_layer_weight(x)
        residual_weight = self.gen_residual_weight(x)

        module_in = jnp.einsum("hbi,h->bi", x, pre_input_weight).reshape(module_input_shape)
        module_out = self.module(module_in).reshape(batch_count, -1)
        module_out = jnp.einsum("bi,h->hbi", module_out, post_layer_weight)
        residual = jnp.einsum("Hbi,Hh->hbi", x, residual_weight)

        return (module_out + residual).reshape(input_shape)

import time
import functools
import typing as T

Args = T.ParamSpec("Args")
Ret = T.TypeVar("Ret")


def CountPerformance(func: T.Callable[Args, Ret]) -> T.Callable[Args, tuple[Ret, float]]:
    @functools.wraps(func)
    def wrapper(*args: Args.args, **kwargs: Args.kwargs) -> tuple[Ret, float]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    return wrapper

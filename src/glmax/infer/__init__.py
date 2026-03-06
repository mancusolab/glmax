# pattern: Imperative Shell


def infer(*args, **kwargs):
    """Canonical infer verb entrypoint."""
    from .inference import infer as _infer

    return _infer(*args, **kwargs)


def check(*args, **kwargs):
    """Canonical check verb entrypoint."""
    from .diagnostics import check as _check

    return _check(*args, **kwargs)


__all__ = ["infer", "check"]

# server/src/core/run_context.py

_RUN_CONTEXT = None


def set_run_context(ctx):
    global _RUN_CONTEXT
    _RUN_CONTEXT = ctx


def get_run_context():
    if _RUN_CONTEXT is None:
        raise RuntimeError("Run context not initialized")
    return _RUN_CONTEXT
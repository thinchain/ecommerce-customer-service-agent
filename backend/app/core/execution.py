"""Per-node audit context, scoped to the async task rather than shared agent state."""
from contextlib import contextmanager
from contextvars import ContextVar

execution_context = ContextVar('execution_context', default={})


@contextmanager
def execution_span(agent, phase, **extra):
    token = execution_context.set({'agent': agent, 'phase': phase, **extra})
    try:
        yield
    finally:
        execution_context.reset(token)

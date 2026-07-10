import anyio

from src.network.transport.anyio_cancel_guard import (
    _patch_deliver_cancellation,
    _unpatch_deliver_cancellation,
)


def test_cancel_guard_does_not_mutate_anyio_scope():
    async def exercise_cancel_scope():
        with anyio.CancelScope() as scope:
            scope.cancel()
            await anyio.sleep(0)
        assert not hasattr(scope, "_oa_cancel_iter")

    try:
        _patch_deliver_cancellation()
        anyio.run(exercise_cancel_scope)
    finally:
        _unpatch_deliver_cancellation()

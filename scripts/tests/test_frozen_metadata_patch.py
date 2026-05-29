"""Regression guard for the frozen-bundle importlib.metadata fix.

Production runtime ``Team`` runs were dying with::

    ERROR Error in Team run: No package metadata was found for pydantic

because ``src.mcp._runtime.function._wrap_callable`` calls
``importlib.metadata.version("pydantic")`` on every Team run and the
bundled ``pydantic-*.dist-info`` directory under ``sys._MEIPASS``
sometimes disappears between launch and that call (sibling-swap, self-
update race, etc.).

``patch_importlib_metadata_for_frozen`` rescues a small allowlist by
falling back to ``module.__version__`` when the dist-info lookup raises.
"""
from __future__ import annotations

import importlib.metadata
import sys

from ._framework import TestContext, test


def _force_frozen(true: bool) -> tuple[object, object]:
    """Toggle ``sys.frozen`` / ``sys._MEIPASS`` and return the prior
    values so the caller can restore them."""
    prior_frozen = getattr(sys, "frozen", None)
    prior_meipass = getattr(sys, "_MEIPASS", None)
    if true:
        sys.frozen = True  # type: ignore[attr-defined]
        sys._MEIPASS = "/tmp"  # type: ignore[attr-defined]
    else:
        if hasattr(sys, "frozen"):
            del sys.frozen  # type: ignore[attr-defined]
        if hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS  # type: ignore[attr-defined]
    return prior_frozen, prior_meipass


def _restore_frozen(prior_frozen: object, prior_meipass: object) -> None:
    if prior_frozen is None:
        if hasattr(sys, "frozen"):
            del sys.frozen  # type: ignore[attr-defined]
    else:
        sys.frozen = prior_frozen  # type: ignore[attr-defined]
    if prior_meipass is None:
        if hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS  # type: ignore[attr-defined]
    else:
        sys._MEIPASS = prior_meipass  # type: ignore[attr-defined]


def _reset_patch_state() -> None:
    """Clear the idempotency sentinel + restore the original ``version``
    function on ``importlib.metadata``. Safe to call even if the patch
    never ran."""
    from src import _frozen

    _frozen._METADATA_PATCHED = False


@test("frozen_metadata_patch", "version() falls back to module.__version__ for pydantic when dist-info is missing")
async def t_version_falls_back_for_pydantic(ctx: TestContext) -> None:
    import pydantic

    from src._frozen import patch_importlib_metadata_for_frozen

    prior = _force_frozen(True)
    orig_version = importlib.metadata.version
    try:
        _reset_patch_state()

        # Simulate a missing dist-info BEFORE the patch installs its
        # wrapper, so the wrapper closes over our raising version().
        def _raising_version(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        importlib.metadata.version = _raising_version  # type: ignore[assignment]
        patch_importlib_metadata_for_frozen()

        result = importlib.metadata.version("pydantic")
        assert result == pydantic.__version__, (
            f"expected fallback to pydantic.__version__={pydantic.__version__!r}, "
            f"got {result!r}"
        )

        # Hyphenated and case-variant names normalize the same way.
        result_hyphen = importlib.metadata.version("Pydantic-Core")
        import pydantic_core
        assert result_hyphen == pydantic_core.__version__, (
            f"name normalization must accept 'Pydantic-Core', got {result_hyphen!r}"
        )
    finally:
        importlib.metadata.version = orig_version  # type: ignore[assignment]
        _reset_patch_state()
        _restore_frozen(*prior)


@test("frozen_metadata_patch", "non-allowlisted names still raise PackageNotFoundError")
async def t_non_allowlisted_still_raises(ctx: TestContext) -> None:
    from src._frozen import patch_importlib_metadata_for_frozen

    prior = _force_frozen(True)
    orig_version = importlib.metadata.version
    try:
        _reset_patch_state()

        def _raising_version(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        importlib.metadata.version = _raising_version  # type: ignore[assignment]
        patch_importlib_metadata_for_frozen()

        raised = False
        try:
            importlib.metadata.version("some_other_package_not_on_the_list")
        except importlib.metadata.PackageNotFoundError:
            raised = True
        assert raised, (
            "version() must still raise PackageNotFoundError for packages "
            "outside the allowlist — the fallback is a narrow safety net"
        )
    finally:
        importlib.metadata.version = orig_version  # type: ignore[assignment]
        _reset_patch_state()
        _restore_frozen(*prior)


@test("frozen_metadata_patch", "patch is idempotent (calling twice does not double-wrap)")
async def t_idempotent(ctx: TestContext) -> None:
    from src._frozen import patch_importlib_metadata_for_frozen

    prior = _force_frozen(True)
    orig_version = importlib.metadata.version
    try:
        _reset_patch_state()
        patch_importlib_metadata_for_frozen()
        wrapped_once = importlib.metadata.version
        patch_importlib_metadata_for_frozen()
        wrapped_twice = importlib.metadata.version
        assert wrapped_once is wrapped_twice, (
            "second call to patch_importlib_metadata_for_frozen must be a "
            "no-op — otherwise repeated startup paths would stack wrappers"
        )
    finally:
        importlib.metadata.version = orig_version  # type: ignore[assignment]
        _reset_patch_state()
        _restore_frozen(*prior)


@test("frozen_metadata_patch", "patch is a no-op when not running frozen")
async def t_noop_when_not_frozen(ctx: TestContext) -> None:
    from src._frozen import patch_importlib_metadata_for_frozen

    prior = _force_frozen(False)
    orig_version = importlib.metadata.version
    try:
        _reset_patch_state()
        patch_importlib_metadata_for_frozen()
        assert importlib.metadata.version is orig_version, (
            "outside a frozen bundle the patch must leave importlib.metadata "
            "untouched so we don't perturb regular pip-installed dev setups"
        )
    finally:
        importlib.metadata.version = orig_version  # type: ignore[assignment]
        _reset_patch_state()
        _restore_frozen(*prior)

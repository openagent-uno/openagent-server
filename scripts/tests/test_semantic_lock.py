"""Recall must not queue behind the indexer's network round trip.

`sync_vault` embeds through the network, and `search` takes the same lock. With
the embed call inside that lock, every recall landing mid-sync waited for the
whole batch to come back. Measured on lyra-agent: the query embedding cost
342ms and the vector scan 420ms, yet recall ran to a 7.4s median, a 25.6s p90
and a 56.6s max — the remainder was purely this wait. Five turns in 76 blew the
30s recall ceiling, and a recall that times out does not raise: it answers the
customer with NO vault rules at all.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ._framework import TestContext, test


class _SlowEmbedder:
    """Stands in for the network hop to the embedding host."""

    model_id = "test-slow"

    def __init__(self, delay: float = 1.0, dim: int = 8):
        self.delay = delay
        self.dim = dim
        self.embedding = threading.Event()

    def embed(self, texts):
        # Only the indexer's multi-item batch is slow. A single text is the
        # recall query embedding itself: keeping it fast is what isolates the
        # lock wait from the query's own cost — the first version of this test
        # measured the stub instead of the lock and blamed the wrong thing.
        if len(texts) > 1:
            self.embedding.set()
            time.sleep(self.delay)
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def _vault(ctx: TestContext, name: str, notes: int) -> tuple[str, str]:
    root = Path(ctx.test_dir) / f"lock-{name}"
    (root / "vault").mkdir(parents=True, exist_ok=True)
    for i in range(notes):
        (root / "vault" / f"n{i}.md").write_text(
            f"---\ntitle: Note {i}\n---\n\nbody about playback {i}\n")
    return str(root / "idx.db"), str(root / "vault")


@test("semantic_lock", "search does not wait for the indexer's embedding round trip")
async def t_search_not_blocked_by_sync(ctx: TestContext) -> None:
    from src.memory.semantic_index import SemanticIndex

    db, vault = _vault(ctx, "a", 6)
    emb = _SlowEmbedder(delay=1.5)
    idx = SemanticIndex(db, vault_root=vault, embedder=emb)
    if not idx.active:
        return  # index inert in this build — nothing to pin

    done = threading.Event()

    def _sync():
        try:
            idx.sync_vault(force=True)
        finally:
            done.set()

    t = threading.Thread(target=_sync, daemon=True)
    t.start()
    # Wait until the indexer is actually inside the embed call.
    assert emb.embedding.wait(timeout=10), "the sync never reached the embedder"

    t0 = time.monotonic()
    idx.search("playback", limit=3)
    elapsed = time.monotonic() - t0

    # The embed call is still in flight here. Before the fix this search blocked
    # for the rest of it; the whole point is that it no longer does.
    assert elapsed < 1.0, (
        f"search waited {elapsed:.2f}s while the indexer was embedding — "
        "the network call is back inside the lock")
    assert not done.is_set(), "the sync finished too early to prove anything"
    t.join(timeout=15)


@test("semantic_lock", "the vectors the indexer embedded still land in the index")
async def t_sync_still_writes(ctx: TestContext) -> None:
    """Moving the embed call out of the lock must not lose the write."""
    from src.memory.semantic_index import SemanticIndex

    db, vault = _vault(ctx, "b", 4)
    emb = _SlowEmbedder(delay=0.0)
    idx = SemanticIndex(db, vault_root=vault, embedder=emb)
    if not idx.active:
        return

    stats = idx.sync_vault(force=True)
    assert stats.embedded == 4, stats
    assert idx.stats()["notes"] == 4, idx.stats()
    # And a second pass is a no-op: mtime/size are unchanged.
    again = idx.sync_vault()
    assert again.embedded == 0, again

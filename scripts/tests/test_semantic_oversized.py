"""One oversized note must cost one note, not the batch it travelled in.

`_MAX_EMBED_CHARS` caps text in CHARACTERS, but an embedding window is
measured in TOKENS, and the exchange rate moves with the tokenizer and the
LANGUAGE — Italian costs more tokens per character than English. Measured on
the eSound vault: a 5,871-char Italian note, under the 6,000 cap, is refused
by `snowflake-arctic-embed2` with "the input length exceeds the context
length".

Before the fix that rejection failed the whole request, so the 31 healthy
notes batched with it went unindexed too — 1,589 such failures on that vault.
An agent that cannot find what it knows answers from nothing.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _PickyEmbedder:
    """Refuses any text over `limit` chars, like a real token window would."""

    def __init__(self, limit: int = 1000, dim: int = 8):
        self.limit = limit
        self.dim = dim
        self.calls: list[int] = []

    def embed(self, texts):
        from src.memory.semantic_index import EmbeddingError

        self.calls.append(len(texts))
        if any(len(t) > self.limit for t in texts):
            raise EmbeddingError("the input length exceeds the context length")
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def _index_with(embedder):
    from src.memory.semantic_index import SemanticIndex

    idx = SemanticIndex.__new__(SemanticIndex)   # no DB / no filesystem needed
    idx.embedder = embedder
    return idx


@test("semantic_oversized", "an oversized item no longer takes its whole batch down")
async def t_batch_survives(ctx: TestContext) -> None:
    emb = _PickyEmbedder(limit=1000)
    idx = _index_with(emb)
    texts = ["short note"] * 20 + ["x" * 5000] + ["short note"] * 11
    out = idx._embed_batch(texts)

    # One vector per input, still aligned with the items.
    assert len(out) == len(texts), (len(out), len(texts))
    dims = {d for _blob, d in out}
    assert len(dims) == 1, f"ragged matrix: {dims}"

    # The batch was retried item by item, not abandoned.
    assert emb.calls[0] == len(texts), emb.calls[:1]
    assert emb.calls.count(1) >= len(texts) - 1


@test("semantic_oversized", "the oversized item is shrunk until it fits")
async def t_shrinks(ctx: TestContext) -> None:
    emb = _PickyEmbedder(limit=1000)
    idx = _index_with(emb)
    out = idx._embed_batch(["y" * 5000])
    assert len(out) == 1
    # Halving 5000 reaches <=1000 in three steps; it must not give up first.
    assert out[0][1] == emb.dim, out


@test("semantic_oversized", "a text refused at every size yields an aligned placeholder")
async def t_hopeless_item(ctx: TestContext) -> None:
    class _AlwaysRefuses(_PickyEmbedder):
        def embed(self, texts):
            from src.memory.semantic_index import EmbeddingError

            self.calls.append(len(texts))
            # The width probe is the one call that succeeds, so the
            # placeholder can be the right shape.
            if len(texts) == 1 and texts[0] == "x":
                return [[1.0] + [0.0] * (self.dim - 1)]
            raise EmbeddingError("nope")

    emb = _AlwaysRefuses()
    idx = _index_with(emb)
    out = idx._embed_batch(["whatever"])
    assert len(out) == 1
    assert out[0][1] == emb.dim, "placeholder must match the endpoint's width"

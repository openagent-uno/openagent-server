"""SentenceChunker — boundaries, abbreviations, decimals, code fences, flush.

The chunker is what stands between an LLM token stream and the TTS API.
A regression here means we either re-narrate a sentence that was split
across a tool call, or we tile a code-block aloud. Both are loud bugs,
hence the broad coverage.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("tts_chunker", "splits plain sentences on '. ' boundaries")
async def t_basic_split(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out = c.feed(
        "Welcome to the demo today. The weather is mild and breezy. "
        "It should rain tomorrow afternoon."
    )
    assert len(out) >= 2, f"expected >= 2 chunks, got {out}"
    assert out[0].endswith("today."), out[0]
    tail = c.flush()
    full = " ".join(out + ([tail] if tail else []))
    assert "afternoon" in full


@test("tts_chunker", "abbreviations (Dr. Smith, e.g.) do not break sentences")
async def t_abbreviations(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out = c.feed("Dr. Smith said hi to the class today.")
    tail = c.flush()
    chunks = out + ([tail] if tail else [])
    assert len(chunks) == 1, f"expected single chunk, got {chunks}"
    assert "Dr. Smith" in chunks[0]

    c2 = SentenceChunker()
    out2 = c2.feed("Use a comma here, e.g. between clauses, please.")
    tail2 = c2.flush()
    chunks2 = out2 + ([tail2] if tail2 else [])
    assert len(chunks2) == 1, f"expected single chunk, got {chunks2}"


@test("tts_chunker", "decimals (3.14) do not break sentences")
async def t_decimals(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out = c.feed("Pi is 3.14 here in this example today.")
    tail = c.flush()
    chunks = out + ([tail] if tail else [])
    assert len(chunks) == 1, f"expected single chunk, got {chunks}"
    assert "3.14" in chunks[0]


@test("tts_chunker", "code fences are skipped, placeholder emitted once")
async def t_code_fences(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    text = (
        "Here is the example function for you. "
        "```python\n"
        "def hi():\n"
        "    return 42\n"
        "```\n"
        "That should run as expected."
    )
    out = c.feed(text)
    tail = c.flush()
    full = " ".join(out + ([tail] if tail else []))
    assert "Code shown on screen" in full, full
    assert "def hi" not in full, full
    assert "return 42" not in full, full
    # placeholder appears exactly once
    assert full.count("Code shown on screen") == 1, full


@test("tts_chunker", "flush() returns trailing partial below MIN_LEN")
async def t_flush_trailing(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    # No terminal punctuation — chunker should hold and flush the lot.
    out = c.feed("Just a tiny tail")
    assert out == [], out
    tail = c.flush()
    assert tail == "Just a tiny tail", tail


@test("tts_chunker", "streaming feed: partial deltas accumulate correctly")
async def t_streaming_feed(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    pieces = [
        "Hello, this is a test of streami",
        "ng deltas. ",
        "Here's the second sentence to come. ",
        "Final one.",
    ]
    collected: list[str] = []
    for p in pieces:
        collected.extend(c.feed(p))
    if tail := c.flush():
        collected.append(tail)
    full = " ".join(collected)
    assert "streaming deltas" in full, full
    assert "Final one" in full, full
    # at least two boundaries fired
    assert len(collected) >= 2, collected


@test("tts_chunker", "iteration_break() forces flush across tool-loop turns")
async def t_iteration_break(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    c.feed("First half of a sentence wrapping ")
    mid = c.iteration_break()
    assert mid == "First half of a sentence wrapping", mid
    # After iteration_break, state is clean — feed continues fresh
    out = c.feed("Second turn starts here, completely new now. ")
    tail = c.flush()
    chunks = out + ([tail] if tail else [])
    assert any("Second turn" in ch for ch in chunks), chunks


@test("tts_chunker", "trailing partial backticks are deferred to next feed")
async def t_partial_fence(ctx: TestContext) -> None:
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out1 = c.feed("Some prose here ``")  # 2 trailing ticks; could become ```
    # Should not have entered fence yet; "Some prose here" is below MIN_LEN
    assert out1 == [], out1
    out2 = c.feed("`code\n```\nDone with the run.")  # closes the fence
    tail = c.flush()
    full = " ".join(out2 + ([tail] if tail else []))
    assert "Code shown on screen" in full, full
    assert "code" not in full or full.count("code") == 0, full
    assert "Done with the run" in full, full

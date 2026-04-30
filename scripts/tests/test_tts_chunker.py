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
    # First-chunk early-emission may split at the leading comma, but
    # the *abbreviation* protection — "e.g." NOT triggering a sentence
    # break — is what this test guards. So no chunk may end at "e.g."
    # (which would mean the period after `g` was misread as a
    # sentence end). Joined together the chunks must still preserve
    # the full text in order.
    joined = " ".join(chunks2)
    assert "e.g." in joined, f"e.g. should be preserved verbatim: {joined}"
    for ch in chunks2:
        assert not ch.rstrip().endswith("e.g."), (
            f"abbreviation 'e.g.' must not fire a sentence break: {ch!r}"
        )


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


@test("tts_chunker", "first chunk emits early at first comma boundary")
async def t_first_chunk_clause_early(ctx: TestContext) -> None:
    """The very first chunk of an iteration uses the relaxed
    boundary set (CLAUSE_PUNCT + MIN_FIRST_LEN) so Piper can start
    synthesising sooner. Without this the user waits for the first
    full sentence which can be 3–5 s on long replies."""
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out = c.feed("Hello there friends, welcome to the demo today.")
    # First chunk should fire at the comma — text up to that point is
    # 19 chars, comfortably above MIN_FIRST_LEN.
    assert len(out) >= 1, out
    assert out[0].endswith(","), f"first chunk should end at clause boundary: {out[0]!r}"
    assert "Hello there friends" in out[0], out[0]


@test("tts_chunker", "second chunk reverts to sentence-only behaviour")
async def t_second_chunk_strict(ctx: TestContext) -> None:
    """After the first chunk emits, clause boundaries should NOT
    split sentences anymore — that's the natural-prosody promise."""
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    # First chunk: "Sure thing," at the comma (TTFB win).
    # Second sentence has internal commas that must NOT split it.
    out = c.feed(
        "Sure thing, here we go. The plan, as we discussed, "
        "involves three steps. Done."
    )
    tail = c.flush()
    chunks = out + ([tail] if tail else [])
    # Find the chunk containing "as we discussed" — it must be the
    # full middle sentence, not a fragment.
    middle = next((ch for ch in chunks if "as we discussed" in ch), None)
    assert middle is not None, f"missing 'as we discussed' chunk in {chunks}"
    assert "three steps" in middle, (
        f"second-chunk path must NOT split at clause comma: {middle!r}"
    )


@test("tts_chunker", "iteration_break re-arms first-chunk early-emission")
async def t_iteration_break_rearms_first(ctx: TestContext) -> None:
    """Each tool-loop iteration deserves its own TTFB win — voice
    mode often pauses for tool calls between sentences and the user
    benefits from a quick spoken status when the agent resumes."""
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    # "Sure thing now," is 14 chars — comfortably above MIN_FIRST_LEN.
    out1 = c.feed("Sure thing now, let me check that.")
    assert any(ch.endswith(",") for ch in out1), (
        f"first chunk of iter 1 should be clause-bounded: {out1}"
    )
    c.iteration_break()
    # "All done now," is 13 chars — also above MIN_FIRST_LEN. The bar
    # for iteration 2 must match iteration 1 (the whole point of
    # re-arming): if iteration_break didn't reset the flag, a comma
    # in this string would be ignored.
    out2 = c.feed("All done now, here are the results.")
    assert any(ch.endswith(",") for ch in out2), (
        f"first chunk of iter 2 should ALSO be clause-bounded: {out2}"
    )


@test("tts_chunker", "tiny first phrase below MIN_FIRST_LEN waits for sentence end")
async def t_first_chunk_too_short_defers(ctx: TestContext) -> None:
    """``"Hi, world."`` should NOT emit ``"Hi,"`` as the first chunk
    (only 3 chars, below MIN_FIRST_LEN). It defers to the period.
    Otherwise we'd ship a 0.2s clip and Piper synth overhead would
    swamp the playback."""
    from openagent.channels.tts_chunker import SentenceChunker

    c = SentenceChunker()
    out = c.feed("Hi, world.")
    tail = c.flush()
    chunks = out + ([tail] if tail else [])
    # Should be exactly one chunk: "Hi, world." — the comma was below
    # the floor, the period closes the sentence.
    assert len(chunks) == 1, f"tiny-phrase comma must defer: {chunks}"
    assert chunks[0].endswith("."), chunks[0]


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

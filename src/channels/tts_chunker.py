"""Sentence chunker for streaming TTS.

Buffers incoming text deltas from the LLM and emits speakable chunks at
sentence boundaries. Keeps abbreviations and decimals from being split,
skips code fences (replaced with a single placeholder per fence), and
defers chunks below ``MIN_LEN`` so we don't ship 0.4-second clips.

The very first chunk of an iteration uses a relaxed boundary set so the
user hears something within ~1 s on Piper instead of waiting for the
first full sentence (which can take 3–5 s for long replies). Concretely:

* terminal punctuation OR clause punctuation (``,``, ``;``, ``:``)
* ``MIN_FIRST_LEN`` (12) instead of ``MIN_LEN`` (20)

Once the first chunk emits, subsequent chunks revert to the strict
sentence-only rules so the bulk of the reply preserves natural prosody.
``iteration_break()`` re-arms the first-chunk mode so each tool-loop
iteration also gets the TTFB win on its first sentence.

Use::

    chunker = SentenceChunker()
    for delta in llm_stream:
        for sentence in chunker.feed(delta):
            await tts.synthesize(sentence)
    if tail := chunker.flush():
        await tts.synthesize(tail)

Across LLM tool-loop iterations, call ``iteration_break()`` instead of
``flush()`` so a sentence that straddles a tool call isn't re-narrated.
"""

from __future__ import annotations


class SentenceChunker:
    MIN_LEN = 20
    # Lower bar for the first emitted chunk so Piper can start
    # synthesising sooner. 12 chars is roughly 2–3 short words —
    # big enough to be worth a synth call, small enough that a
    # quick "Sure, let me…" greeting fires immediately.
    MIN_FIRST_LEN = 12
    PLACEHOLDER = "Code shown on screen."
    ABBREVIATIONS: tuple[str, ...] = (
        "Mr.", "Mrs.", "Ms.", "Dr.", "Sr.", "Jr.", "St.", "Mt.",
        "e.g.", "i.e.", "etc.", "vs.", "Prof.", "Inc.", "Ltd.", "Co.",
    )
    # Clause boundaries used ONLY for the first-chunk path. Adding
    # these to the steady-state boundary set butchers prosody (commas
    # mid-sentence don't carry the same intonation drop a period does),
    # so we restrict them to the time-to-first-audio optimisation.
    CLAUSE_PUNCT: tuple[str, ...] = (",", ";", ":")

    def __init__(self) -> None:
        self._raw = ""
        self._speakable = ""
        self._in_fence = False
        self._placeholder_pending = False
        # When False, ``_next_chunk`` accepts CLAUSE_PUNCT boundaries
        # and uses MIN_FIRST_LEN. Flipped to True after the first
        # successful emission. Reset by ``iteration_break`` so each
        # tool-loop turn gets its own TTFB win.
        self._first_chunk_emitted = False

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._raw += delta
        self._consume_fences()
        return self._extract_sentences()

    def flush(self) -> str | None:
        if self._in_fence and self._placeholder_pending:
            self._speakable += " " + self.PLACEHOLDER + " "
            self._placeholder_pending = False
        self._in_fence = False
        if self._raw:
            self._speakable += self._raw
            self._raw = ""
        text = " ".join(self._speakable.split()).strip()
        self._speakable = ""
        return text or None

    def iteration_break(self) -> str | None:
        # Re-arm the first-chunk early-emission mode so the next
        # iteration's first sentence also gets the TTFB win, not just
        # the very first sentence of the entire reply.
        result = self.flush()
        self._first_chunk_emitted = False
        return result

    # ── internals ──────────────────────────────────────────────────────

    def _consume_fences(self) -> None:
        # Defer trailing 1-2 backtick partials — they may complete a ``` next feed.
        partial = ""
        n = len(self._raw)
        if n > 0 and self._raw[-1] == "`":
            run = 1
            while run < n and run < 3 and self._raw[-1 - run] == "`":
                run += 1
            if run < 3:
                partial = self._raw[-run:]
                self._raw = self._raw[:-run]
        work = self._raw
        self._raw = partial

        i, m = 0, len(work)
        out: list[str] = []
        while i < m:
            if work[i:i + 3] == "```":
                if self._in_fence:
                    self._in_fence = False
                    if self._placeholder_pending:
                        out.append(" " + self.PLACEHOLDER + " ")
                        self._placeholder_pending = False
                else:
                    self._in_fence = True
                    self._placeholder_pending = True
                i += 3
                continue
            if not self._in_fence:
                out.append(work[i])
            i += 1

        if out:
            self._speakable += "".join(out)

    def _extract_sentences(self) -> list[str]:
        chunks: list[str] = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def _next_chunk(self) -> str | None:
        text = self._speakable
        if not text:
            return None
        # First-chunk mode lowers the bar in two ways: clause punct
        # counts as a boundary AND the min length drops. Both reset
        # to strict sentence behaviour after the first emission.
        in_first_mode = not self._first_chunk_emitted
        min_len = self.MIN_FIRST_LEN if in_first_mode else self.MIN_LEN
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch == "\n" and i + 1 < n and text[i + 1] == "\n":
                chunk = text[:i].strip()
                if len(chunk) >= min_len:
                    self._speakable = text[i + 2:].lstrip()
                    self._first_chunk_emitted = True
                    return chunk
                i += 2
                continue
            if ch in ".!?…":
                if i + 1 < n and text[i + 1] in " \n\t":
                    if self._is_abbreviation(text, i):
                        i += 1
                        continue
                    chunk = text[:i + 1].strip()
                    if len(chunk) >= min_len:
                        self._speakable = text[i + 1:].lstrip()
                        self._first_chunk_emitted = True
                        return chunk
            elif in_first_mode and ch in self.CLAUSE_PUNCT:
                # Clause-boundary path: only fires until the first
                # successful emission. Same followed-by-whitespace
                # rule as terminal punct so we don't split inside a
                # decimal/URL/etc.
                if i + 1 < n and text[i + 1] in " \n\t":
                    chunk = text[:i + 1].strip()
                    if len(chunk) >= min_len:
                        self._speakable = text[i + 1:].lstrip()
                        self._first_chunk_emitted = True
                        return chunk
            i += 1
        return None

    def _is_abbreviation(self, text: str, dot_idx: int) -> bool:
        j = dot_idx - 1
        while j >= 0 and (text[j].isalpha() or text[j] == "."):
            j -= 1
        token = text[j + 1:dot_idx + 1]
        return any(token.endswith(a) for a in self.ABBREVIATIONS)

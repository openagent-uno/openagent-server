from __future__ import annotations

import unittest

from src.stream.content_parts import (
    ContentMarkerStreamFilter,
    UiMarkerStreamFilter,
    parse_response_content,
    scrub_run_output_carriers_for_storage,
)


class CustomUiContentPartsTests(unittest.TestCase):
    def test_preserves_text_attachment_and_ui_order(self):
        parsed = parse_response_content(
            "Before [FILE:/tmp/report.pdf] middle "
            "[OPENAGENT_UI:status-board@3] after",
            allow_inline_ui=True,
        )
        self.assertEqual(parsed.text, "Before  middle  after")
        self.assertEqual(
            [part["kind"] for part in parsed.parts],
            ["text", "attachment", "text", "ui_view", "text"],
        )
        self.assertEqual(parsed.parts[3]["view_id"], "status-board")
        self.assertEqual(parsed.parts[3]["revision"], 3)

    def test_text_only_surface_strips_ui_marker(self):
        parsed = parse_response_content(
            "Dashboard created [OPENAGENT_UI:board@1]",
            allow_inline_ui=False,
        )
        self.assertEqual(parsed.text, "Dashboard created")
        self.assertNotIn("ui_view", [part["kind"] for part in parsed.parts])
        self.assertEqual(parsed.ui_refs[0]["view_id"], "board")

    def test_malformed_ui_carrier_never_leaks(self):
        parsed = parse_response_content(
            "Safe fallback [OPENAGENT_UI:not valid@x]",
            allow_inline_ui=True,
        )
        self.assertEqual(parsed.text, "Safe fallback")

    def test_malformed_attachment_carrier_never_leaks(self):
        parsed = parse_response_content(
            "Safe fallback [FILE:]",
            allow_inline_ui=False,
        )
        self.assertEqual(parsed.text, "Safe fallback")

    def test_stream_filter_hides_marker_split_across_deltas(self):
        filt = UiMarkerStreamFilter()
        visible = "".join((
            filt.feed("Here is [OPEN"),
            filt.feed("AGENT_UI:board"),
            filt.feed("@12] your live view"),
            filt.finish(),
        ))
        self.assertEqual(visible, "Here is  your live view")
        self.assertEqual(filt.refs, [{
            "kind": "ui_view", "view_id": "board", "revision": 12,
        }])

    def test_stream_filter_releases_non_marker_partial_prefix(self):
        filt = UiMarkerStreamFilter()
        visible = filt.feed("Use [OPEN") + filt.finish()
        self.assertEqual(visible, "Use [OPEN")

    def test_stream_filter_hides_every_legacy_marker_at_every_split(self):
        carriers = (
            "[FILE:/tmp/report.pdf]",
            "[IMAGE:/tmp/photo.png]",
            "[VOICE:/tmp/reply.ogg]",
            "[VIDEO:/tmp/demo.mp4]",
            "[OPENAGENT_UI:board@9]",
        )
        for carrier in carriers:
            for split in range(1, len(carrier)):
                filt = ContentMarkerStreamFilter()
                visible = "".join((
                    filt.feed("before " + carrier[:split]),
                    filt.feed(carrier[split:] + " after"),
                    filt.finish(),
                ))
                self.assertEqual(visible, "before  after", (carrier, split, visible))

    def test_stream_filter_keeps_oversized_carrier_hidden_until_close(self):
        filt = ContentMarkerStreamFilter(max_marker_chars=16)
        visible = "".join((
            filt.feed("safe [FILE:" + "x" * 40),
            filt.feed("still-hidden"),
            filt.feed("] visible"),
            filt.finish(),
        ))
        self.assertEqual(visible, "safe  visible")

    def test_storage_copy_drops_carriers_without_mutating_live_output(self):
        from types import SimpleNamespace

        live_message = SimpleNamespace(
            role="assistant",
            content="See [FILE:/tmp/report.pdf] [OPENAGENT_UI:board@2] now",
        )
        live = SimpleNamespace(
            content="Ready [OPENAGENT_UI:board@2]",
            messages=[live_message],
        )
        import copy

        stored = copy.copy(live)
        scrub_run_output_carriers_for_storage(stored)
        self.assertEqual(stored.content, "Ready")
        self.assertEqual(stored.messages[0].content, "See   now")
        self.assertEqual(live.content, "Ready [OPENAGENT_UI:board@2]")
        self.assertEqual(
            live.messages[0].content,
            "See [FILE:/tmp/report.pdf] [OPENAGENT_UI:board@2] now",
        )


if __name__ == "__main__":
    unittest.main()

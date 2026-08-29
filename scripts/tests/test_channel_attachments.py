"""Structured attachment transport across bridge and MCP boundaries."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ._framework import TestContext, test


@test("channel_attachments", "BaseBridge sends inbound attachments and author on TextFinal")
async def t_bridge_inbound_wire(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge

    bridge = BaseBridge.__new__(BaseBridge)
    bridge.name = "probe"
    bridge._stream_opened = set()
    bridge._stream_pending = {}
    bridge._ws = object()
    sent: list[dict] = []

    async def _capture(payload):
        sent.append(payload)

    bridge._send_gateway_json = _capture  # type: ignore[method-assign]
    author = {"kind": "human", "handle": "telegram:42", "display": "Ada"}
    attachment = {
        "type": "image", "path": "/tmp/photo.jpg", "filename": "photo.jpg",
        "mime_type": "image/jpeg",
    }

    async def _resolve():
        while "tg:42" not in bridge._stream_pending:
            await asyncio.sleep(0)
        collector = bridge._stream_pending["tg:42"]
        collector.text = "ok"
        collector.done.set()

    await asyncio.gather(
        bridge.send_message(
            "look", "tg:42", attachments=[attachment], author=author,
        ),
        _resolve(),
    )
    final = next(item for item in sent if item["type"] == "text_final")
    assert final["attachments"] == [attachment]
    assert final["author"] == author


@test("channel_attachments", "live bridge replies still deliver native AttachmentRefs")
async def t_bridge_live_outbound_attachment(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge

    delivered = []
    chunks: list[str] = []

    class _Stub(BaseBridge):
        name = "probe"

        async def send_text_chunk(self, target, chunk):
            chunks.append(chunk)

        async def send_attachment(self, target, attachment):
            delivered.append(attachment)

    bridge = _Stub.__new__(_Stub)
    bridge.name = "probe"
    bridge._live = True
    with tempfile.TemporaryDirectory(prefix="oa-native-att-") as raw:
        path = Path(raw) / "report.pdf"
        path.write_bytes(b"%PDF-probe")

        async def _reply(text, session_id, **kwargs):
            await kwargs["on_delta"]("Before tool. ")
            await kwargs["on_status"](json.dumps({"tool_name": "bash"}))
            return {
                "type": "response",
                "text": "Before tool. Done.",
                "accumulated": "Before tool. Done.",
                "model": None,
                "attachments": [{
                    "artifact_id": "art_probe",
                    "type": "file",
                    "path": str(path),
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                }],
                "target": None,
            }

        bridge.send_message = _reply  # type: ignore[method-assign]
        await bridge.dispatch_turn("target", "sid", "question")
        assert [item.filename for item in delivered] == ["report.pdf"]
        assert delivered[0].mime_type == "application/pdf"
        assert any("Before tool" in chunk for chunk in chunks)


@test("channel_attachments", "non-live bridge replies deliver native AttachmentRefs once")
async def t_bridge_non_live_outbound_attachment(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge

    delivered = []
    chunks: list[str] = []

    class _Stub(BaseBridge):
        name = "probe"

        async def send_text_chunk(self, target, chunk):
            chunks.append(chunk)

        async def send_attachment(self, target, attachment):
            delivered.append(attachment)

    bridge = _Stub.__new__(_Stub)
    bridge.name = "probe"
    bridge._live = False
    with tempfile.TemporaryDirectory(prefix="oa-native-att-off-") as raw:
        path = Path(raw) / "report.pdf"
        path.write_bytes(b"%PDF-probe")

        async def _reply(text, session_id, **kwargs):
            return {
                "type": "response",
                "text": "Done.",
                "accumulated": "Done.",
                "model": None,
                "attachments": [{
                    "artifact_id": "art_probe",
                    "type": "file",
                    "path": str(path),
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                }],
                "target": None,
            }

        bridge.send_message = _reply  # type: ignore[method-assign]
        await bridge.dispatch_turn("target", "sid", "question")
        assert chunks == ["Done."], chunks
        assert [item.filename for item in delivered] == ["report.pdf"]


@test("channel_attachments", "Telegram extracts typed photo and image-document refs safely")
async def t_telegram_extract_structured(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge

    class _Media:
        def __init__(self, unique_id, payload, **fields):
            self.file_unique_id = unique_id
            self.file_size = len(payload)
            self.payload = payload
            for key, value in fields.items():
                setattr(self, key, value)

        async def get_file(self):
            return self

        async def download_to_drive(self, path):
            Path(path).write_bytes(self.payload)

    class _Message:
        photo = [_Media("photo-id", b"jpeg")]
        voice = None
        audio = None
        video = None
        video_note = None
        document = _Media(
            "doc-id", b"\x89PNG\r\n\x1a\n",
            file_name="../../diagram.png", mime_type="image/png",
        )

        async def reply_text(self, text):
            raise AssertionError(text)

    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge.name = "telegram"
    with tempfile.TemporaryDirectory(prefix="oa-tg-extract-") as raw:
        extracted = await bridge._extract_files(_Message(), raw)
        assert [item["type"] for item in extracted.attachments] == ["image", "image"]
        assert extracted.attachments[1]["filename"] == "diagram.png"
        assert all(Path(item["path"]).parent == Path(raw).resolve()
                   for item in extracted.attachments)
        assert len({item["path"] for item in extracted.attachments}) == 2


@test("channel_attachments", "Telegram PDF and JSON captions reach durable CAS refs, not path text")
async def t_telegram_documents_reach_cas(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge
    from src.memory.artifacts import (
        artifact_row,
        normalize_inbound_attachments,
        public_attachment_ref,
    )

    from .test_artifacts import _access, _artifact_db, _seed_session

    class _Media:
        def __init__(self, unique_id, payload, *, filename, mime_type):
            self.file_unique_id = unique_id
            self.file_id = f"remote-{unique_id}"
            self.file_size = len(payload)
            self.file_name = filename
            self.mime_type = mime_type
            self.payload = payload

        async def get_file(self):
            return self

        async def download_to_drive(self, path):
            Path(path).write_bytes(self.payload)

    class _User:
        id = 7
        username = "ada"
        full_name = "Ada"

    class _Message:
        photo = None
        voice = None
        audio = None
        video = None
        video_note = None
        text = None
        reply_to_message = None
        media_group_id = None
        chat = SimpleNamespace(type="private", id=7)
        from_user = _User()

        def __init__(self, caption, document):
            self.caption = caption
            self.document = document

        async def reply_text(self, text):
            raise AssertionError(text)

    class _Update:
        def __init__(self, message):
            self.message = message

    pdf_bytes = b"%PDF-1.4\ntelegram-pdf"
    json_bytes = b'{"source":"telegram-json"}'
    messages = [
        _Message(
            "read the quarterly report",
            _Media(
                "pdf-id", pdf_bytes,
                filename="quarterly.pdf", mime_type="application/pdf",
            ),
        ),
        _Message(
            "inspect this payload",
            _Media(
                "json-id", json_bytes,
                filename="payload.json", mime_type="application/json",
            ),
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="oa-tg-cas-") as raw:
        db, tenant = _artifact_db(Path(raw))
        _seed_session(db, tenant, "tg:7", "ada")
        bridge = TelegramBridge.__new__(TelegramBridge)
        bridge.name = "telegram"
        bridge.allowed_chats = None
        bridge._bot_username = None
        bridge._is_fresh_update = lambda _update: True  # type: ignore[method-assign]
        bridge._is_authorized = lambda _uid: True  # type: ignore[method-assign]
        captured: list[dict] = []

        async def _dispatch(target, session_id, text, **kwargs):
            canonical = await normalize_inbound_attachments(
                db,
                kwargs["attachments"],
                session_id=session_id,
                principal=_access(tenant, "ada"),
                allow_local_paths=True,
            )
            captured.append({
                "text": text,
                "attachments": canonical,
                "author": kwargs["author"],
            })

        bridge.dispatch_turn = _dispatch  # type: ignore[method-assign]
        for message in messages:
            await bridge._on_message(_Update(message), SimpleNamespace())

        assert [item["text"] for item in captured] == [
            "read the quarterly report",
            "inspect this payload",
        ]
        assert all("local path" not in item["text"] for item in captured)
        refs = [item["attachments"][0] for item in captured]
        assert [ref["type"] for ref in refs] == ["file", "file"]
        assert [ref["mime_type"] for ref in refs] == [
            "application/pdf", "application/json",
        ]
        assert all(ref["artifact_id"].startswith("art_") for ref in refs)
        assert all(ref["url"].endswith("/content") for ref in refs)
        assert all("path" not in public_attachment_ref(ref) for ref in refs)
        assert captured[0]["author"]["handle"] == "telegram:7"

        stored = [await artifact_row(db, ref["artifact_id"]) for ref in refs]
        assert [path.read_bytes() for _row, path in stored] == [pdf_bytes, json_bytes]
        assert all("artifacts/sha256" in str(path) for _row, path in stored)


@test("channel_attachments", "Telegram missing and duplicate document names stay distinct and CAS-linked")
async def t_telegram_document_names_and_cas(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge
    from src.memory.artifacts import normalize_inbound_attachments, public_attachment_ref

    from .test_artifacts import _access, _artifact_db, _seed_session

    class _Media:
        def __init__(
            self, payload, *, unique_id, file_id, filename, mime_type,
        ):
            self.payload = payload
            self.file_unique_id = unique_id
            self.file_id = file_id
            self.file_size = len(payload)
            self.file_name = filename
            self.mime_type = mime_type

        async def get_file(self):
            return self

        async def download_to_drive(self, path):
            Path(path).write_bytes(self.payload)

    class _Message:
        photo = None
        voice = None
        audio = None
        video = None
        video_note = None

        def __init__(self, document):
            self.document = document

        async def reply_text(self, text):
            raise AssertionError(text)

    duplicate_bytes = b"%PDF-duplicate"
    media = [
        _Media(
            duplicate_bytes, unique_id="same-id", file_id="remote-a",
            filename="report.pdf", mime_type="application/pdf",
        ),
        _Media(
            duplicate_bytes, unique_id="same-id", file_id="remote-b",
            filename="report.pdf", mime_type="application/pdf",
        ),
        _Media(
            b'{"missing":"name"}', unique_id=None, file_id="fallback-id",
            filename=None, mime_type="application/json",
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="oa-tg-names-") as raw:
        root = Path(raw)
        stage = root / "stage"
        stage.mkdir()
        db, tenant = _artifact_db(root)
        _seed_session(db, tenant, "tg:7", "ada")
        bridge = TelegramBridge.__new__(TelegramBridge)
        bridge.name = "telegram"

        extracted = [
            await bridge._extract_files(_Message(item), str(stage))
            for item in media
        ]
        staged = [item.attachments[0] for item in extracted]
        assert [item["filename"] for item in staged] == [
            "report.pdf", "report.pdf", "document_fallback-id",
        ]
        assert len({item["path"] for item in staged}) == 3
        assert [Path(item["path"]).read_bytes() for item in staged] == [
            duplicate_bytes, duplicate_bytes, b'{"missing":"name"}',
        ]

        refs = await normalize_inbound_attachments(
            db,
            staged,
            session_id="tg:7",
            principal=_access(tenant, "ada"),
            allow_local_paths=True,
        )
        assert refs[0]["artifact_id"] == refs[1]["artifact_id"]
        assert refs[0]["artifact_link_id"] != refs[1]["artifact_link_id"]
        assert refs[2]["artifact_id"] != refs[0]["artifact_id"]
        assert all("path" not in public_attachment_ref(ref) for ref in refs)


@test("channel_attachments", "Telegram album keeps group session and all typed attachments")
async def t_telegram_album_group_session(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge, _Extracted

    class _User:
        id = 7
        username = "ada"
        full_name = "Ada"

    class _Message:
        from_user = _User()
        chat = SimpleNamespace(type="supergroup", id=-42)
        text = None
        reply_to_message = None

        def __init__(self, caption, token):
            self.caption = caption
            self.token = token

    messages = [_Message("@openagent inspect", "a"), _Message(None, "b")]
    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge.name = "telegram"
    bridge._bot_username = "openagent"
    bridge._media_group_lock = asyncio.Lock()
    key = ("tg:group:-42", "album-1")
    bridge._media_groups = {key: {"messages": messages, "timer": None}}
    captured: dict = {}

    async def _extract(message, tmp):
        return _Extracted(attachments=[{
            "type": "image",
            "path": str(Path(tmp) / f"{message.token}.jpg"),
            "filename": f"{message.token}.jpg",
            "mime_type": "image/jpeg",
        }])

    async def _dispatch(target, session_id, text, **kwargs):
        captured.update(session_id=session_id, text=text, **kwargs)

    bridge._extract_files = _extract  # type: ignore[method-assign]
    bridge.dispatch_turn = _dispatch  # type: ignore[method-assign]
    await bridge._flush_media_group(key)
    assert captured["session_id"] == "tg:group:-42"
    assert len(captured["attachments"]) == 2
    assert captured["text"] == "inspect"
    assert all(item["path"] not in captured["text"] for item in captured["attachments"])
    assert captured["author"]["handle"] == "telegram:7"


@test("channel_attachments", "Telegram shutdown cancels and awaits every album timer")
async def t_telegram_album_timer_shutdown(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge

    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge._media_group_lock = asyncio.Lock()
    bridge._media_groups = {}
    bridge._media_group_tasks = set()
    finished: list[str] = []

    async def _pending(name: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(name)

    tasks = [asyncio.create_task(_pending(str(index))) for index in range(3)]
    await asyncio.sleep(0)
    bridge._media_group_tasks.update(tasks)
    bridge._media_groups[("tg:1", "album")] = {
        "messages": [object()],
        "timer": tasks[-1],
    }

    await bridge._cancel_media_group_timers()

    assert not bridge._media_groups
    assert all(task.done() for task in tasks), tasks
    assert sorted(finished) == ["0", "1", "2"], finished


@test("channel_attachments", "Telegram rejects pre/post-download oversize and uses NAME_MAX-safe staging")
async def t_telegram_post_download_limit_and_staging(ctx: TestContext) -> None:
    import src.bridges.telegram as telegram_mod
    from src.bridges.telegram import TelegramBridge

    class _Media:
        file_unique_id = "📎" * 200
        file_size = 1

        def __init__(self, payload: bytes):
            self.payload = payload

        async def get_file(self):
            return self

        async def download_to_drive(self, path):
            Path(path).write_bytes(self.payload)

    class _Message:
        def __init__(self):
            self.replies: list[str] = []

        async def reply_text(self, text):
            self.replies.append(text)

    bridge = TelegramBridge.__new__(TelegramBridge)
    message = _Message()
    with tempfile.TemporaryDirectory(prefix="oa-tg-limit-") as raw:
        attempted_downloads: list[str] = []

        class _DeclaredOversize(_Media):
            file_size = 4

            async def get_file(self):
                attempted_downloads.append("downloaded")
                return self

        with patch.object(telegram_mod, "attachment_limit_bytes", return_value=3):
            pre_rejected = await bridge._download_media(
                message,
                _DeclaredOversize(b"four"),
                raw,
                filename="declared-large.pdf",
                kind="file",
                mime_type="application/pdf",
            )
        assert pre_rejected is None
        assert attempted_downloads == []
        assert not list(Path(raw).iterdir())

        with patch.object(telegram_mod, "attachment_limit_bytes", return_value=3):
            rejected = await bridge._download_media(
                message,
                _Media(b"four"),
                raw,
                filename=("界" * 200) + ".pdf",
                kind="file",
                mime_type="application/pdf",
            )
        assert rejected is None
        assert any("troppo grande" in reply for reply in message.replies), message.replies
        assert not list(Path(raw).iterdir())

        with patch.object(telegram_mod, "attachment_limit_bytes", return_value=0):
            accepted = await bridge._download_media(
                message,
                _Media(b"ok"),
                raw,
                filename=("界" * 200) + ".pdf",
                kind="file",
                mime_type="application/pdf",
            )
        assert accepted is not None
        assert len(Path(accepted["path"]).name.encode("utf-8")) <= 240

        class _BrokenMedia(_Media):
            async def get_file(self):
                raise OSError("telegram unavailable")

        failed = await bridge._download_media(
            message,
            _BrokenMedia(b""),
            raw,
            filename="failed.pdf",
            kind="file",
            mime_type="application/pdf",
        )
        assert failed is None
        assert any("Impossibile scaricare" in reply for reply in message.replies)


@test("channel_attachments", "Slack is an activatable bridge")
async def t_slack_selected(ctx: TestContext) -> None:
    from src.core.server import _selected_bridge_names

    config = {"channels": {"slack": {"bot_token": "x", "app_token": "y"}}}
    assert _selected_bridge_names(config, None) == ["slack"]
    assert _selected_bridge_names(config, ["gateway", "slack"]) == ["slack"]


@test("channel_attachments", "Slack scopes channels jointly and DMs per user")
async def t_slack_session_scope(ctx: TestContext) -> None:
    from src.bridges.slack import SlackBridge

    assert SlackBridge._session_id("U1", "C42") == "sl:channel:C42"
    assert SlackBridge._session_id("U2", "C42") == "sl:channel:C42"
    assert SlackBridge._session_id("U1", "D11") == "sl:U1"
    assert SlackBridge._session_id("U2", "D11", channel_type="im") == "sl:U2"


@test("channel_attachments", "Slack downloader reports limits and keeps staging basename safe")
async def t_slack_download_outcome_and_staging(ctx: TestContext) -> None:
    import src.bridges.slack as slack_mod
    from src.bridges.slack import SlackBridge

    bridge = SlackBridge.__new__(SlackBridge)
    bridge.bot_token = "xoxb-test"

    # Pin a tiny cap so no HTTP session is needed for the rejection path.
    with patch.object(slack_mod, "attachment_limit_bytes", return_value=3):
        too_large = await bridge._download_file(
            "https://files.slack.com/report.pdf",
            tempfile.gettempdir(),
            "report.pdf",
            declared_size=4,
        )
    assert too_large.path is None and too_large.error == "too_large"

    class _Content:
        async def iter_chunked(self, _size):
            yield b"ok"

    class _Response:
        status = 200
        headers = {}
        content = _Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class _Session:
        def get(self, *args, **kwargs):
            return _Response()

    async def _session():
        return _Session()

    bridge._audio_session = _session  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory(prefix="oa-sl-name-") as raw:
        with patch.object(slack_mod, "attachment_limit_bytes", return_value=0):
            outcome = await bridge._download_file(
                "https://files.slack.com/" + ("p" * 300),
                raw,
                ("界" * 200) + ".pdf",
            )
        assert outcome.path is not None
        assert len(Path(outcome.path).name.encode("utf-8")) <= 240


@test("channel_attachments", "Slack inbound, slash and picker keep scope and report oversize")
async def t_slack_handlers_scope_and_notifications(ctx: TestContext) -> None:
    from src.bridges.slack import SlackBridge, _SlackDownloadOutcome

    handlers: dict[str, object] = {}
    notices: list[dict] = []

    class _Client:
        async def auth_test(self):
            return {"user_id": "UBOT"}

        async def chat_postMessage(self, **kwargs):
            notices.append(kwargs)

    class _AsyncApp:
        def __init__(self, token):
            self.client = _Client()

        def event(self, name):
            def _decorator(fn):
                handlers[f"event:{name}"] = fn
                return fn
            return _decorator

        def command(self, name):
            def _decorator(fn):
                handlers[f"command:{name}"] = fn
                return fn
            return _decorator

        def action(self, name):
            def _decorator(fn):
                handlers[f"action:{name}"] = fn
                return fn
            return _decorator

    class _SocketHandler:
        def __init__(self, app, token):
            self.app = app

        async def start_async(self):
            return None

    module_names = (
        "slack_bolt",
        "slack_bolt.async_app",
        "slack_bolt.adapter",
        "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.async_handler",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["slack_bolt.async_app"].AsyncApp = _AsyncApp
    modules["slack_bolt.adapter.socket_mode.async_handler"].AsyncSocketModeHandler = (
        _SocketHandler
    )

    bridge = SlackBridge(
        bot_token="xoxb-test",
        app_token="xapp-test",
        allowed_users=["U1", "U2"],
        listen_channels=["C42"],
    )
    with patch.dict(sys.modules, modules):
        await bridge._run()

    async def _oversize(*args, **kwargs):
        return _SlackDownloadOutcome(path=None, error="too_large", size_bytes=9)

    dispatched: list[str] = []

    async def _dispatch(target, session_id, text, **kwargs):
        dispatched.append(session_id)

    bridge._download_file = _oversize  # type: ignore[method-assign]
    bridge.dispatch_turn = _dispatch  # type: ignore[method-assign]
    on_message = handlers["event:message"]
    await on_message(  # type: ignore[operator]
        {
            "user": "U1",
            "channel": "C42",
            "channel_type": "channel",
            "text": "caption",
            "files": [{
                "id": "F1",
                "name": "large.pdf",
                "url_private_download": "https://files.slack.com/large.pdf",
                "size": 1,
            }],
        },
        None,
        {},
        bridge._app.client,
    )

    async def _failed(*args, **kwargs):
        return _SlackDownloadOutcome(path=None, error="download_failed")

    bridge._download_file = _failed  # type: ignore[method-assign]
    await on_message(  # type: ignore[operator]
        {
            "user": "U1",
            "channel": "C42",
            "channel_type": "channel",
            "text": "caption two",
            "files": [{
                "id": "F2",
                "name": "missing.pdf",
                "url_private_download": "https://files.slack.com/missing.pdf",
            }],
        },
        None,
        {},
        bridge._app.client,
    )
    await on_message(  # type: ignore[operator]
        {
            "user": "U2",
            "channel": "D22",
            "channel_type": "im",
            "text": "private",
        },
        None,
        {},
        bridge._app.client,
    )
    assert dispatched == ["sl:channel:C42", "sl:channel:C42", "sl:U2"], dispatched
    assert any("troppo grande" in item["text"] for item in notices), notices
    assert any("Impossibile scaricare" in item["text"] for item in notices), notices

    command_sessions: list[str] = []

    async def _command_full(command, *, session_id, arg=None):
        command_sessions.append(session_id)
        return {"text": "done", "picker": None}

    async def _command(command, *, session_id, arg=None):
        command_sessions.append(session_id)
        return "done"

    async def _ack():
        return None

    async def _respond(*args, **kwargs):
        return None

    bridge.send_command_full = _command_full  # type: ignore[method-assign]
    bridge.send_command = _command  # type: ignore[method-assign]
    await handlers["command:/model"](  # type: ignore[operator]
        _ack,
        {"user_id": "U1", "channel_id": "C42", "text": "auto"},
        _respond,
    )
    await handlers["action:oa_cmd_pick"](  # type: ignore[operator]
        _ack,
        {
            "user": {"id": "U2"},
            "channel": {"id": "C42"},
            "actions": [{"selected_option": {"value": "model::auto"}}],
        },
        _respond,
    )
    assert command_sessions == ["sl:channel:C42", "sl:channel:C42"]


@test("channel_attachments", "WhatsApp audio preserves source filename and MIME")
async def t_whatsapp_audio_metadata(ctx: TestContext) -> None:
    from src.bridges.whatsapp import WhatsAppBridge

    bridge = WhatsAppBridge.__new__(WhatsAppBridge)
    bridge.name = "whatsapp"
    bridge.allowed_users = None
    captured: dict = {}
    requested_names: list[str] = []

    async def _download(url, filename):
        requested_names.append(filename)
        tmp = Path(tempfile.mkdtemp(prefix="oa-wa-probe-"))
        path = tmp / filename
        path.write_bytes(b"m4a-probe")
        return str(path)

    async def _transcribe(path):
        return "voice transcript"

    async def _dispatch(target, session_id, text, **kwargs):
        captured.update(target=target, session_id=session_id, text=text, **kwargs)

    bridge._download = _download  # type: ignore[method-assign]
    bridge.transcribe_with_fallback = _transcribe  # type: ignore[method-assign]
    bridge.dispatch_turn = _dispatch  # type: ignore[method-assign]
    await bridge._handle({
        "senderData": {
            "chatId": "391234567890@c.us",
            "sender": "391234567890@c.us",
            "senderName": "Ada",
        },
        "messageData": {
            "typeMessage": "audioMessage",
            "fileMessageData": {
                "downloadUrl": "https://example.invalid/audio",
                "fileName": "meeting.m4a",
                "mimeType": "audio/mp4",
            },
        },
    })
    assert requested_names == ["meeting.m4a"]
    assert captured["voice_detected"] is True
    assert captured["attachments"][0]["filename"] == "meeting.m4a"
    assert captured["attachments"][0]["mime_type"] == "audio/mp4"
    assert captured["author"]["handle"] == "whatsapp:391234567890"


@test("channel_attachments", "WhatsApp blocks SSRF literals, DNS and redirects")
async def t_whatsapp_download_ssrf_guards(ctx: TestContext) -> None:
    import aiohttp
    from src.bridges.whatsapp import (
        WhatsAppBridge,
        _PublicAddressResolver,
        _WhatsAppDownloadError,
        _validate_whatsapp_media_url,
    )

    unsafe = (
        "http://files.example/report.pdf",
        "https://127.0.0.1/report.pdf",
        "https://169.254.169.254/latest/meta-data",
        "https://user:pass@files.example/report.pdf",
        "https://files.example:8443/report.pdf",
    )
    for url in unsafe:
        try:
            _validate_whatsapp_media_url(url)
        except _WhatsAppDownloadError:
            pass
        else:
            raise AssertionError(f"unsafe URL accepted: {url}")

    resolver = _PublicAddressResolver()
    loop = asyncio.get_running_loop()
    private_answer = [
        (2, 1, 6, "", ("10.0.0.7", 443)),
    ]
    with patch.object(loop, "getaddrinfo", return_value=private_answer):
        try:
            await resolver.resolve("files.example", 443)
        except OSError:
            pass
        else:
            raise AssertionError("private DNS answer was accepted")

    class _RedirectResponse:
        status = 302
        headers = {"Location": "https://127.0.0.1/secret"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            return _RedirectResponse()

    bridge = WhatsAppBridge.__new__(WhatsAppBridge)
    with (
        patch.object(aiohttp, "TCPConnector", return_value=object()),
        patch.object(aiohttp, "ClientSession", return_value=_Session()),
    ):
        try:
            await bridge._download("https://files.example/start", "report.pdf")
        except _WhatsAppDownloadError:
            pass
        else:
            raise AssertionError("redirect to loopback was accepted")


@test("channel_attachments", "WhatsApp notifies download and post-size failures")
async def t_whatsapp_download_failure_notices(ctx: TestContext) -> None:
    from src.bridges.whatsapp import (
        WhatsAppBridge,
        _WhatsAppAttachmentTooLarge,
        _WhatsAppDownloadError,
    )

    bridge = WhatsAppBridge.__new__(WhatsAppBridge)
    sent: list[str] = []

    async def _send(chat_id, text):
        sent.append(text)

    async def _too_large(url, filename):
        raise _WhatsAppAttachmentTooLarge(99)

    bridge._send_text = _send  # type: ignore[method-assign]
    bridge._download = _too_large  # type: ignore[method-assign]
    assert await bridge._download_or_notify("chat", "https://x", "big.pdf") is None
    assert "troppo grande" in sent[-1]

    async def _failed(url, filename):
        raise _WhatsAppDownloadError("network")

    bridge._download = _failed  # type: ignore[method-assign]
    assert await bridge._download_or_notify("chat", "https://x", "bad.pdf") is None
    assert "Impossibile scaricare" in sent[-1]


@test("channel_attachments", "MCP audio and embedded blobs become typed tool media")
async def t_mcp_typed_media(ctx: TestContext) -> None:
    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        Tool,
    )
    from src.core._runner.utils.mcp import get_entrypoint_for_tool

    class _Session:
        async def send_ping(self):
            return None

        async def call_tool(self, name, kwargs, meta=None):
            return CallToolResult(
                isError=False,
                content=[
                    AudioContent(
                        type="audio",
                        data=base64.b64encode(b"audio-bytes").decode(),
                        mimeType="audio/ogg",
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri="file:///report.pdf",
                            mimeType="application/pdf",
                            blob=base64.b64encode(b"%PDF-probe").decode(),
                        ),
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri="file:///clip.mp4",
                            mimeType="video/mp4",
                            blob=base64.b64encode(b"video-bytes").decode(),
                        ),
                    ),
                ],
            )

    entrypoint = get_entrypoint_for_tool(
        Tool(name="media_probe", description="probe", inputSchema={}),
        _Session(),  # type: ignore[arg-type]
    )
    result = await entrypoint()
    assert result.audios and result.audios[0].get_content_bytes() == b"audio-bytes"
    assert result.files and result.files[0].get_content_bytes() == b"%PDF-probe"
    assert result.files[0].filename == "report.pdf"
    assert result.videos and result.videos[0].get_content_bytes() == b"video-bytes"

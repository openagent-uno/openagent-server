#!/usr/bin/env python3
"""Scored, zero-business-I/O replay of multi-turn support regressions.

All business MCPs are in-memory simulators. The ONLY optional external process
is a tool-less model adapter: JSON {system,messages} in, {content,model} out.
No production database is cloned, no scheduler is started, no pending events
are inherited. Exit nonzero on wrong routing, invented evidence or unsafe calls.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.core import local_support_controller as controller
from src.core.dry_run import dry_run_scope
from src.core.support_turn import requested_fields
from scripts.tests.test_local_support_controller import _Doubles


@dataclass(frozen=True)
class Case:
    id: str
    product: str
    turns: tuple[tuple[str, str], ...]
    intent: str
    reader_kind: str
    forbidden_requests: tuple[str, ...] = ()
    must_call: tuple[str, ...] = ()
    must_not_call: tuple[str, ...] = ()
    forbidden_reply: tuple[str, ...] = ()
    expected_outcome: str = ""
    image_fixture: str = ""
    expected_image_text: str = ""
    incoming_message: str = ""
    channel: str = "email_imap"
    receipt_turns: tuple[int, ...] = ()
    contracted_latest: bool = False
    required_reply: tuple[str, ...] = ()


CASES = (
    Case("post51-version-followup-pt", "esound", (
        ("inbound", "não consigo mais baixar musica"),
        ("outbound", "Para investigar isso, envie a versão do aplicativo."),
        ("inbound", "4.15.10.16")), "bug", "bug", forbidden_requests=("app_version",),
        forbidden_reply=("Premium não permite", "catálogo")),
    Case("post51-device-burst-pt", "esound", (
        ("inbound", "não consigo mais baixar musica"),
        ("outbound", "Para investigar isso, envie a versão do aplicativo."),
        ("inbound", "4.15.10.16"),
        ("outbound", "eSound transmite seu catálogo: Premium não permite downloads do catálogo."),
        ("inbound", "o problema é que tbm nao da para ouvir musicas novas tbm"),
        ("outbound", "Para investigar isto, envie o dispositivo e o sistema operativo."),
        ("inbound", "abcdef0123456789abcdef0123456789"),
        ("inbound", "iphone11")), "bug", "bug", forbidden_requests=("app_version", "device"),
        forbidden_reply=("catálogo",), incoming_message="abcdef0123456789abcdef0123456789",
        contracted_latest=True),
    Case("post51-short-version", "lyra", (
        ("inbound", "The player crashes whenever I tap Play."),
        ("outbound", "Which app version are you using?"),
        ("inbound", "5.2.4")), "bug", "bug", forbidden_requests=("app_version",)),
    Case("post51-auto-receipt-is-not-answer", "esound", (
        ("inbound", "The app crashes whenever I open my playlist."),
        ("outbound", "Thanks for your comment! Our team has received it and will get back to you shortly.")),
        "bug", "bug", incoming_message="The app crashes whenever I open my playlist.",
        must_call=("replio_threads_respond",), receipt_turns=(1,), channel="instagram_dm"),
    Case("audit-explicit-refund-de", "esound", (("inbound", "Ich möchte eine Rückerstattung für meine letzte Premium-Zahlung."),),
         "refund", "refund_request", expected_outcome="refund_identity_required",
         forbidden_reply=("I need", "Please send", "erstattet habe")),
    Case("audit-signup-failure-pt", "lyra", (("inbound", "Estou tentando criar uma conta, mas depois de preencher o email e tocar em Criar conta aparece um erro e a conta não é criada."),),
         "bug", "bug", forbidden_reply=("abra o aplicativo", "toque em criar conta para começar")),

    Case("audit-account-recovery-es", "esound", (("inbound", "Quiero recuperar mi cuenta"),),
         "account_change", "account_recovery", forbidden_requests=("receipt",),
         forbidden_reply=("voy a cambiar", "procesarlo", "reembolso", "recibo"),
         expected_outcome="account_change_identity_required"),
    Case("audit-account-change-confirm-es", "esound", (
         ("inbound", "Quiero cambiar el correo de mi perfil."),
         ("outbound", "Voy a cambiar el correo de tu perfil. Déjame procesarlo."),
         ("inbound", "Sí, puedes hacerlo.")),
         "account_change", "account_change", forbidden_reply=("voy a cambiar", "he cambiado", "procesarlo"),
         expected_outcome="account_change_identity_required"),
    Case("audit-conditional-refund-it", "esound", (("inbound",
         "L'app si blocca quando apro una canzone. Se no sarò costretto a chiedere il rimborso. Android 15, Samsung S21, app 5.2.4."),),
         "bug", "bug", must_not_call=("billingbear_refund",), forbidden_reply=("ricevuta", "ordine di acquisto")),
    Case("audit-apple-notice", "esound", (("inbound", "We have completed our review of your submission."),),
         "machine_mail", "", must_not_call=("replio_threads_respond",), expected_outcome="machine_mail"),
    Case("audit-business-enquiry", "lyra", (("inbound", "We propose a business partnership with your company. Can we schedule a call about our advertising service?"),),
         "business_request", "business_request", must_call=("replio_threads_mark_for_human",),
         must_not_call=("replio_threads_respond",), expected_outcome="business_request_human"),
    Case("audit-ad-frequency-review", "esound", (("inbound", "There are way too many ads. An ad every two songs is unbearable."),),
         "premium", "ads_feedback", forbidden_requests=("app_version", "device", "steps"),
         expected_outcome="ads_policy_explained", channel="playstore_reviews",
         required_reply=("server", "referral", "video")),
    Case("audit-ad-overlay-bug", "esound", (("inbound", "After an ad finishes its overlay stays on the screen and its audio keeps playing over my music. Android 15, Samsung S21, version 5.2.4."),),
         "bug", "bug", forbidden_reply=("upgrade to premium",)),
    Case("audit-missing-logout-pt", "lyra", (
         ("inbound", "Como faço para trocar de conta?"),
         ("outbound", "Abra as configurações e toque em Sair para fazer login com outra conta."),
         ("inbound", "Não existe botão Sair nas configurações. Já procurei lá. Esse menu não existe no meu app.")),
         "guidance_question", "", forbidden_reply=("toque em sair", "tap log out", "clique em sair"),
         expected_outcome="guidance_unavailable_human", must_call=("replio_threads_mark_for_human",)),

    Case("private-channel-stays-here", "lyra", (
        ("inbound", "Can I message you in DMs? I do not want to put my email here."),),
        "support_channel", "support_channel", expected_outcome="support_channel_answer",
        forbidden_reply=("email us", "main support channel", "secure", "what problem"), channel="instagram_dm"),
    Case("public-channel-invites-dm", "lyra", (
        ("inbound", "Could we talk in DMs? This comment is too long."),),
        "support_channel", "support_channel", expected_outcome="support_channel_answer",
        forbidden_reply=("post your email", "email us", "I have sent"), channel="instagram_comment"),
    Case("positive-portuguese-review", "esound", (
        ("inbound", "Muito bom, adorei o aplicativo, excelente para ouvir música!"),),
        "praise", "", expected_outcome="praise_thanks",
        forbidden_reply=("?", "problema", "versão", "device"), channel="playstore_reviews"),
    Case("bot-stop-persists-on-image", "esound", (
        ("inbound", "Here are the account details you already asked for."),
        ("outbound", "Please send your account email and receipt."),
        ("inbound", "Bot don't reply"),
        ("inbound", "[1 attachment(s): image]")),
        "human_request", "", expected_outcome="human_requested_no_reply",
        must_call=("replio_threads_mark_for_human",), must_not_call=("replio_threads_respond",), channel="messenger"),
    Case("media-burst-explanation-survives", "esound", (
         ("inbound", "When I search and play a song, a different song plays instead.\n---\napp_version: 5.2.4\ndevice: Samsung S21\nos: Android 15\nplatform: android"),
         ("outbound", "Try playing the song from a fresh search."),
         ("inbound", "[1 attachment(s): video]"),
         ("inbound", "Now it gives me an AI voice explaining the song instead of playing it."),
         ("inbound", "Is this a prank?")),
         "bug", "bug", ("app_version", "device", "steps"),
         forbidden_reply=("describe the attachment", "send it again"),
         incoming_message="[1 attachment(s): video]"),
    Case("ios-availability-is-not-a-bug-questionnaire", "esound", (
         ("inbound", "Are you available on iOS?"),
         ("outbound", "What happens in the app? Send your device and version."),
         ("inbound", "dawg nothing is happening, i just switched from android to ios on the same account and i just asked if its available on ios thats all")),
         "guidance_question", "guidance_question", ("app_version", "device", "steps"),
         must_not_call=("clickup_create_task",), expected_outcome="guidance_answer"),
    Case("resolved-playback-it", "esound", (
         ("inbound", "La musica non parte nell'app desktop."),
         ("outbound", "Installa l'ultima versione e prova una canzone."),
         ("inbound", "Buongiorno, ho disinstallato la vecchia versione e installato l'ultima, ora la musica è partita")),
         "resolved_confirmation", "resolved_confirmation", must_not_call=("replio_threads_respond", "clickup_create_task"),
         expected_outcome="resolved_confirmation"),
    Case("resolved-correction-it", "esound", (
         ("inbound", "Ora la musica è partita."),
         ("outbound", "Quale dispositivo e versione dell'app usi?"),
         ("inbound", "Ho detto che ora l'app desktop funziona perfettamente, la musica è partita regolarmente")),
         "resolved_confirmation", "resolved_confirmation", must_not_call=("replio_threads_respond", "clickup_create_task"),
         expected_outcome="resolved_confirmation"),
    Case("thanks-with-unmarked-quote", "esound", (
         ("inbound", "My folders are empty when I try adding songs."),
         ("outbound", "Folders hold playlists and albums, not individual songs. Put the songs into a playlist first, then move that playlist into the folder."),
         ("inbound", "So many thanks!\nI will follow your instructions later.\nCheers\n\nFolders hold playlists and albums, not individual songs. Put the songs into a playlist first, then move that playlist into the folder.")),
         "acknowledgement", "acknowledgement", must_not_call=("replio_threads_respond", "clickup_create_task"),
         expected_outcome="acknowledgement_no_reply_needed"),
    Case("log-explanation-followup-it", "esound", (
         ("inbound", "La riproduzione si blocca su Android 15, versione 5.2.4, Samsung S21."),
         ("outbound", "Mandaci un log o una registrazione dello schermo."),
         ("inbound", "Non so cos’è un log e come trovarlo.")),
         "guidance_question", "guidance_question", ("app_version", "device"),
         must_not_call=("clickup_create_task",),
         forbidden_reply=("support review", "versione?"), expected_outcome="guidance_answer"),
    Case("recording-instruction-followup-en", "lyra", (
         ("inbound", "Playback stops on my Samsung S21, Android 15, app 1.4.11."),
         ("outbound", "Please send a screen recording."),
         ("inbound", "I don't know how to do that.")),
         "guidance_question", "guidance_question", ("app_version", "device"),
         must_not_call=("clickup_create_task",), expected_outcome="guidance_answer"),
    Case("signup-pt", "esound", (("inbound", "Quero fazer minha conta"),),
         "account_signup", "signup", must_not_call=("clickup_create_task", "esound_identity_delete_account")),
    Case("catalog-download-es", "esound", (("inbound", "Quiero descargar canciones para escuchar sin internet."),),
         "offline", "catalog_offline", must_not_call=("clickup_create_task",),
         forbidden_reply=("con la suscripción premium", "compra premium")),
    Case("lost-downloads-en", "esound", (("inbound",
         "Three songs have disappeared from my library and all my playlists. I downloaded them from the app's catalog. "
         "They no longer appear in offline songs; I can only find them by searching with internet."),),
         "bug", "library_loss", must_call=("clickup_create_task",),
         forbidden_reply=("streams its catalog", "reinstall", "nothing is lost")),
    Case("lost-library-pt", "lyra", (("inbound",
         "Minhas músicas salvas sumiram da biblioteca depois da atualização. Antes eu ouvia offline."),),
         "bug", "library_loss", must_call=("clickup_create_task",),
         forbidden_reply=("premium não", "reinstale")),
    Case("referral-followup-pt", "lyra", (
         ("inbound", "Indiquei um amigo hoje, mas a indicação não apareceu.\n---\napp_version: 1.4.11\ndevice: Samsung S21\nos: Android 15"),
         ("outbound", "Qual foi o passo a passo?"),
         ("inbound", "Eu compartilhei o código de indicação."),
         ("outbound", "O que você faz no app e o que espera?"),
         ("inbound", "A pessoa digitou o código no app e enviou. Espero que apareça nas minhas indicações para ganhar 30 dias.")),
         "referral_status", "referral_status", ("steps", "device", "app_version"),
         ("replio_threads_mark_for_human",), ("clickup_create_task",),
         ("onde exatamente", "já foi creditado", "aguarde 3 dias")),
    Case("native-startup-followup", "lyra", (
         ("inbound", "Lyra crashes immediately. I have a Samsung S21+ running Android 15, app 1.4.11."),
         ("outbound", "Does it crash when you tap the icon or while loading?"),
         ("inbound", "I tap the icon, it loads for a second, but crashes before it can show any UI.")),
         "bug", "bug", ("steps", "device", "app_version"),
         ("clickup_create_task",), (), ("keep the app open", "reinstall", "nothing is lost")),
    Case("status-followup-it", "lyra", (
         ("inbound", "L'app si blocca all'avvio su Android 15, Samsung S21, versione 1.4.11."),
         ("outbound", "Ci stiamo guardando."), ("inbound", "Avete sistemato?")),
         "status_check", "status_check", ("steps", "device", "app_version"),
         ("replio_threads_mark_for_human",), (), ("è stato risolto", "reinstalla")),
    Case("password-reset-is-not-signup", "esound", (("inbound", "I forgot my password. How can I reset it?"),),
         "password_recovery", "password_recovery", must_not_call=("esound_identity_delete_account",),
         forbidden_reply=("we can send", "i'll send", "send you a reset", "i'll need the email"),
         expected_outcome="password_recovery_self_service"),
    Case("free-premium-is-not-referral-incident", "lyra", (("inbound", "How do I get Premium for free without paying for the ads?"),),
         "premium", "", must_not_call=("clickup_create_task",),
         forbidden_reply=("isn't a free way", "receipt", "order id", "email"),
         expected_outcome="ads_policy_explained"),
    Case("topic-change-does-not-revive-old-bug", "esound", (
         ("inbound", "My library disappeared on Android 15, app 5.2.0."),
         ("outbound", "Your library issue is fixed."),
         ("inbound", "Different question now: how can I download catalog music to listen offline?")),
         "offline", "catalog_offline", must_not_call=("clickup_create_task",)),
    Case("vision-error-screenshot", "esound", (("inbound", ""),), "bug", "",
         must_call=("replio_thread_read_attachment",),
         must_not_call=("clickup_create_task",),
         forbidden_reply=("cannot read", "describe the attachment", "send it again"),
         image_fixture="support-vision-wc037.png", expected_image_text="WC037"),
)


class StdioModel:
    def __init__(self, command: list[str]):
        self.command = command
        self.calls: list[dict[str, Any]] = []
        self.override = ""

    def build_override_model(self, spec):
        model = StdioModel(self.command)
        model.calls = self.calls
        model.override = spec
        return model

    async def generate(self, *, messages, system, session_id, images=None):
        proc = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await proc.communicate(json.dumps({
                "messages": messages, "system": system,
                "model": self.override,
                "images": [{"mime_type": img.mime_type, "data": base64.b64encode(img.content).decode()}
                           for img in images or []],
            }).encode())
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"model adapter exited {proc.returncode}")
        result = json.loads(stdout)
        self.calls.append({"leg": session_id.rsplit(":", 1)[-1], "model": result.get("model"),
                           "content": result.get("content")})
        return SimpleNamespace(content=result["content"])


def score(case: Case, output: dict[str, Any], doubles: _Doubles) -> list[str]:
    failures = []
    if "replio_threads_respond" not in case.must_not_call:
        if (not output.get("reply") or
                output.get("facts", {}).get("reply_source") != "model:human_voice_verified" or
                "replio_threads_respond" not in doubles.names):
            failures.append("expected a reviewed contextual reply to be delivered")
    if case.contracted_latest and output.get("facts", {}).get("message_source") != "latest_contracted_inbound":
        failures.append("queued text was not aligned with the verified latest inbound")
    if output["intent"] != case.intent:
        failures.append(f"intent: expected {case.intent}, got {output['intent']}")
    if case.reader_kind and output.get("customer_reported", {}).get("kind") != case.reader_kind:
        failures.append("conversation reader did not recover expected request")
    if case.expected_outcome and output.get("outcome") != case.expected_outcome:
        failures.append("wrong resolution: " + str(output.get("outcome")))
    if case.image_fixture and not output.get("facts", {}).get("attachment_readable"):
        failures.append("image was not actually interpreted")
    if case.expected_image_text and case.expected_image_text not in output.get("attachment_evidence", {}).get("visible_text", ""):
        failures.append("visible image text did not match ground truth")
    requests = requested_fields(output.get("reply", ""))
    if requests.intersection(case.forbidden_requests):
        failures.append("asked again for provided fields: " + str(sorted(requests.intersection(case.forbidden_requests))))
    for name in case.must_call:
        if name not in doubles.names:
            failures.append("missing operation: " + name)
    for name in case.must_not_call:
        if name in doubles.names:
            failures.append("forbidden operation: " + name)
    for text in case.forbidden_reply:
        if text.casefold() in output.get("reply", "").casefold():
            failures.append("forbidden reply claim: " + text)
    for text in case.required_reply:
        equivalents = {"referral": ("referr", "inviting friends", "invite friends"),
                       "server": ("server", "infrastructure", "running costs")}.get(text, (text.casefold(),))
        if not any(term in output.get("reply", "").casefold() for term in equivalents):
            failures.append("missing useful answer content: " + text)
    if output["facts"].get("delivery_state") not in {"simulated", "not_attempted"}:
        failures.append("delivery not explicitly simulated/noop")
    for action in output["actions"]:
        if (action.get("success") and action.get("kind") not in {"task_link_verify", "diagnostic_read"}
                and not action.get("receipt", {}).get("simulated")):
            failures.append("successful mutation without simulation receipt")
    return failures


async def replay(command: list[str], selected: list[Case], repeat: int) -> dict[str, Any]:
    os.environ.update({"OPENAGENT_FORCE_DRY_RUN": "1",
                       "OPENAGENT_SUPPORT_HUMAN_VOICE": "1",
                       "OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES": "1",
                       "OPENAGENT_SUPPORT_TURN_READER": "1",
                       # The registered model is the only external dependency.
                       "OPENAGENT_SUPPORT_SEMANTIC_ROUTING": "0"})
    rows = []
    for iteration in range(repeat):
        for case in selected:
            model = StdioModel(command)
            attachment = None
            if case.image_fixture:
                raw = (Path(__file__).parent / "fixtures" / case.image_fixture).read_bytes()
                attachment = {"content": [{"type": "image", "mimeType": "image/png",
                                           "data": base64.b64encode(raw).decode()}]}
            messages = [
                {"direction": direction, "body_text": body,
                 "external_message_id": f"fixture-{i}",
                 "counts_as_answer": i not in case.receipt_turns}
                for i, (direction, body) in enumerate(case.turns)
            ]
            thread = {"product": case.product, "messages": messages}
            if case.contracted_latest:
                thread["reply_contract"] = {"expected_last_inbound_message_id": messages[-1]["external_message_id"]}
            doubles = _Doubles(thread=thread, attachment=attachment)
            started = time.monotonic()
            try:
                with dry_run_scope(True):
                    result = await controller.run(
                        agent=SimpleNamespace(_mcp=doubles.pool(), model=model),
                        event={"slug": "replio-thread"},
                        payload={"payload": {"thread_id": "sim-" + case.id, "product": case.product,
                            "channel_kind": case.channel, "message": {"body_text": case.incoming_message or case.turns[-1][1],
                            **({"attachments": [{"name": case.image_fixture}]} if case.image_fixture else {})}}},
                        session_id=f"replay:{case.id}:{iteration}", delivery_id="simulated",
                    )
                output = json.loads(result.text)
                failures = score(case, output, doubles)
            except Exception as exc:
                output = {}
                failures = [type(exc).__name__ + ": " + str(exc)[:200]]
            row = {"case": case.id, "iteration": iteration, "failures": failures,
                   "seconds": round(time.monotonic()-started, 2), "output": output,
                   "model_calls": model.calls, "tools": doubles.names}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ["case", "iteration", "failures", "seconds"]}), flush=True)
    return {"cases": len(rows), "passed": sum(not r["failures"] for r in rows),
            "business_io": "in-memory simulators only", "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-command-file", required=True, help="JSON array of model adapter argv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vision-model", default="", help="Registered vision model, independent of the composer")
    args = parser.parse_args()
    command = json.loads(Path(args.model_command_file).read_text())
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        parser.error("model command must be a nonempty string array")
    selected = [case for case in CASES if not args.case or case.id in args.case]
    if args.vision_model:
        os.environ["OPENAGENT_SUPPORT_VISION_MODEL"] = args.vision_model
    if not selected or not 1 <= args.repeat <= 10:
        parser.error("choose at least one case and repeat between 1 and 10")
    report = asyncio.run(replay(command, selected, args.repeat))
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"{report['passed']}/{report['cases']} passed")
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

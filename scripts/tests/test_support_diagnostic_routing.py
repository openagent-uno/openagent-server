"""Symptom -> diagnostic-category routing: the four ways it captured the wrong subsystem."""
from __future__ import annotations

from ._framework import TestContext, test
from src.core import local_support_controller as c


# Byte-for-byte the shape the eSound web form appends to every report: a
# blank line, `---`, then one `key: value` per line.
_FORM_TRAILER = (
    "\n\n---\naccount_email: user@example.com\n"
    "account_user_id: 7f99d3b67b4ce85868a8a123dbbf4fa5\napp_version: 5.2.6\n"
    "device: samsung SM-S918B\nguest: no\ninstall_source: com.android.vending\n"
    "native_version: 5.2.6\nos: Android 14\npackage: com.esound\n"
    "platform: android\npremium: yes\nruntime: native\ntablet: no"
)
# The store-review importer uses the same separator with its own fields.
_REVIEW_TRAILER = (
    "\n\n---\napp_version: 5.2.6\napp_version_code: 834\nos: Android 11\n"
    "device: Samsung a30 (Galaxy A30)\nram_mb: 3746\nreviewer_language: en"
)


@test("support_diagnostic_routing", "device metadata never decides the capture category")
async def t_metadata_does_not_route(_ctx: TestContext):
    symptom = "le canzoni si bloccavano dopo qualche secondo dall inizio dei brani"
    # `premium: yes` rides in the trailer of EVERY report from a paying user.
    # Routing on it captured the billing subsystem for a playback fault.
    assert c._diagnostic_category(symptom + _FORM_TRAILER) == "playback"
    assert c._diagnostic_category(symptom) == "playback"
    # A store review's trailer carries reviewer_language, not premium.
    assert c._diagnostic_category("songs don't load" + _REVIEW_TRAILER) == "playback"
    # The trailer is removed, not merely ignored by the router.
    assert c._strip_report_metadata(symptom + _FORM_TRAILER) == symptom
    # A channel that flattened the trailer onto one line is still a trailer.
    assert c._strip_report_metadata("ciao --- app_version: 5.2.6 premium: yes") == "ciao"
    # `---` the customer typed is NOT a trailer: no known field follows it, so
    # every word they wrote survives.
    for prose in (
        "primo punto\n---\nsecondo punto: importante",
        "il piano premium --- quello annuale --- non si attiva",
    ):
        assert c._strip_report_metadata(prose) == prose, prose


@test("support_diagnostic_routing", "stems match at a word boundary, not mid-word")
async def t_no_midword_matches(_ctx: TestContext):
    # "ad " lived inside "downloa|d |": every download report captured `ads`.
    assert c._diagnostic_category("I can't download my songs anymore") == "playback"
    assert c._diagnostic_category("the app won't load anything") == "general"
    # Italian `ad` is the euphonic preposition, not an advert.
    assert c._diagnostic_category("non riesco ad accedere al mio account") == "auth"
    # Advertising still routes.
    for text in ("I see too many ads", "advertisement everywhere", "troppa pubblicita"):
        assert c._diagnostic_category(text) == "ads", text


@test("support_diagnostic_routing", "specific routes win over the ones that contain them")
async def t_route_order(_ctx: TestContext):
    # `play` is a prefix of `playlist`; with the generic route first, no
    # playlist report could ever reach the playlists capture.
    assert c._diagnostic_category("my playlist is empty") == "playlists"
    assert c._diagnostic_category("la playlist non si apre") == "playlists"
    assert c._diagnostic_category("the player stops") == "playback"


@test("support_diagnostic_routing", "the phrasings customers actually use reach playback")
async def t_playback_vocabulary(_ctx: TestContext):
    # None of these carried one of the four English words the old list had, so
    # they fell to `general` -- which skips the capture entirely.
    for text in (
        "le canzoni si bloccano",
        "la riproduzione si interrompe",
        "songs keep stopping while I listen",
        "a musica para sozinha",
        "la cancion se corta",
        "le son se coupe pendant l ecoute",
    ):
        assert c._diagnostic_category(text) == "playback", text


@test("support_diagnostic_routing", "a playback capture does not send the customer to reproduce on demand")
async def t_playback_bundle_and_instruction(_ctx: TestContext):
    available = {"ads", "general", "network", "playback", "player", "purchases"}
    # The bundle explains the primary category instead of slicing it.
    assert c._diagnostic_bundle("playback", available) == [
        "playback", "player", "network", "general",
    ]
    assert c._diagnostic_bundle("purchases", available) == [
        "purchases", "network", "general",
    ]
    assert len(c._diagnostic_bundle("playback", available)) <= c._MAX_DIAGNOSTIC_CATEGORIES


@test("support_diagnostic_routing", "accented Spanish and Portuguese reach their category")
async def t_accent_folding(_ctx: TestContext):
    # Written unaccented in the stem list; without folding these matched
    # nothing and fell to `general`, which skips the capture outright.
    for text in (
        "no me carga la música",
        "la reproducción se detiene sola",
        "a reprodução para sozinha",
        "la canción se corta",
        "le son se coupe pendant l'écoute",
    ):
        assert c._diagnostic_category(text) == "playback", text
    assert c._fold_accents("reprodução") == "reproducao"


@test("support_diagnostic_routing", "an explicit billing word outranks the word account")
async def t_auth_vs_purchases(_ctx: TestContext):
    # "account" is the commonest word in BOTH kinds of report, so it must not
    # pull a refund thread away from purchases.
    assert c._diagnostic_category("voglio un rimborso, il mio account premium") == "purchases"
    assert c._diagnostic_category("quiero cancelar mi suscripcion") == "purchases"
    # ...and a plain "I can't get back into my account" is authentication,
    # even when the customer mentions their music in the same breath.
    assert c._diagnostic_category(
        "Non riesco a rientrare dentro il mio account di eSound. Ho tutta la musica."
    ) == "auth"
    assert c._diagnostic_category("no puedo entrar en mi cuenta") == "auth"


def _state(outcome: str, **facts) -> c.SupportState:
    s = c.SupportState(thread_id="t", customer_message="search finds nothing")
    s.outcome = outcome
    s.facts.update({"language": "en", **facts})
    return s


@test("support_diagnostic_routing", "a partial capture is not reported as no capture")
async def t_category_missing_is_not_not_captured(_ctx: TestContext):
    # These two outcomes shared one sentence. `not_captured` means zero streams;
    # `category_missing` means the capture is alive and uploading and only the
    # slice tied to the customer's action has not been produced yet. Saying
    # "I cannot find the diagnostics" for the second is false: on 14 set 2026 it
    # went to a customer whose four streams were uploading at that moment, and
    # the missing slice arrived three minutes later carrying the exact proof.
    missing = c._fallback_reply_base(_state("bug_diagnostics_category_missing"))
    none_at_all = c._fallback_reply_base(_state("bug_diagnostics_not_captured"))
    assert missing != none_at_all
    for absolute in ("cannot find", "can't find", "not seeing", "no diagnostic"):
        assert absolute not in missing.lower(), absolute
    # It must ask for the ONE action, after a restart - not the whole sequence.
    assert "reopen" in missing.lower()
    assert "one action" in missing.lower()
    # The Italian branch carries the same meaning, not the old "non trovo".
    it = c._fallback_reply_base(_state("bug_diagnostics_category_missing", language="it"))
    assert "non trovo" not in it.lower()
    assert "riaprire" in it.lower() or "riaperto" in it.lower()


@test("support_diagnostic_routing", "the partial-capture reply claims nothing the turn cannot back")
async def t_category_missing_makes_no_unbacked_claim(_ctx: TestContext):
    # This turn holds no `diagnostic_enable` receipt - it only listed streams -
    # and that receipt is the only envelope either reply guard accepts behind a
    # diagnostics claim. So the reply must instruct, never report capture state.
    for lang in ("en", "it"):
        reply = c._fallback_reply_base(_state("bug_diagnostics_category_missing", language=lang))
        assert not c.reply_guard.claims_diagnostics(reply), lang


@test("support_diagnostic_routing", "the capture is armed before the reproduction, not after")
async def t_enable_suffix_restart_first(_ctx: TestContext):
    # Enabling sets the capture server-side; the client arms it on its NEXT
    # config fetch. "Reproduce, THEN bring the app to the front" therefore armed
    # it with the reproduction already behind it - measured on a real thread:
    # enabled 12:19, customer done 12:25, client armed 12:26:40, nothing caught.
    for italian, reopen, reproduce in (
        (False, "reopen it", "reproduce the issue"),
        (True, "riaprila", "riproduci il problema"),
    ):
        s = c.SupportState(thread_id="t", customer_message="x")
        s.facts["diagnostic_capture"] = {"category": "search", "status": "enabled"}
        text = c._diagnostic_reply_suffix(s, italian)
        assert reopen in text and reproduce in text, text
        assert text.index(reopen) < text.index(reproduce), text
        # Still a diagnostics claim - and rightly so: this turn DOES hold the
        # diagnostic_enable receipt. Losing that would silently unguard it.
        assert c.reply_guard.claims_diagnostics(text)


@test("support_diagnostic_routing", "the web form's own questions never name the broken component")
async def t_form_questions_do_not_route_the_task(_ctx: TestContext):
    # eSound thread eee46c2b, 2-set-2026: a Premium subscriber could not play
    # his playlists - shuffle dead, no next track - and the task was filed as
    # "Fix failure in purchase state" under backend-core. The customer never
    # wrote about buying anything: the form prepends its OWN questions to every
    # report, and "Hai gia' acquistato Premium?" matched the purchase row first.
    report = (
        "Hai già acquistato Premium?: Sì\n"
        "Dove hai acquistato Premium?: Google Play\n\n"
        "Buongiorno mi sono abbonato lo scorso marzo.\n"
        "Nell'ultimo mese non riesco in nessun modo a riprodurre le mie playlist "
        "nonostante la connessione efficiente.\n"
        "Nello specifico non funziona shuffle e non fa cambiare brano."
    ) + _FORM_TRAILER
    route = c._bug_symptom_route(report)
    assert route is not None
    title, _list_id, tag = route
    assert "purchase" not in title, title
    assert title.endswith("in playback"), title
    assert tag == "esound/client-core", tag
    # Stripping the questions must not strip a customer who really IS writing
    # about a purchase.
    billing = c._bug_symptom_route(
        "Hai già acquistato Premium?: Sì\n\nHo pagato il premium ma non funziona" + _FORM_TRAILER
    )
    assert billing is not None and "purchase state" in billing[0], billing
    # English form, same shape.
    en = c._bug_symptom_route(
        "Have you already purchased Premium?: Yes\nWhere did you purchase Premium?: App Store\n\n"
        "Songs are not working, the next track never starts" + _FORM_TRAILER
    )
    assert en is not None and "purchase" not in en[0], en


@test("support_diagnostic_routing", "the web form's own questions never pick the capture category")
async def t_form_questions_do_not_route_the_capture(_ctx: TestContext):
    # The task route already skipped "Hai gia' acquistato Premium?"; the capture
    # route did not, so EVERY web-form playback report captured `purchases`.
    questions = (
        "Hai già acquistato Premium?: Sì\nDove hai acquistato Premium?: Google Play\n\n"
    )
    assert c._diagnostic_category(
        questions + "i brani sono lenti a caricare" + _FORM_TRAILER) == "playback"
    # A customer really writing about a purchase still routes there.
    assert c._diagnostic_category(
        questions + "Ho pagato il premium ma non si attiva" + _FORM_TRAILER) == "purchases"


@test("support_diagnostic_routing", "being a subscriber does not outrank the symptom")
async def t_subscriber_status_loses_to_playback(_ctx: TestContext):
    # eSound, 18-set-2026: "carica per 12 secondi prima di riprodurre, ho pagato
    # un premium" captured `purchases`; the player logs that held the answer
    # were never switched on.
    assert c._diagnostic_category(
        "quando procedo nella riproduzione brani carica per 12 secondi prima di "
        "riprodurre, ho pagato un premium e vorrei che risolvessimo in fretta"
    ) == "playback"
    assert c._diagnostic_category("I'm premium but songs stop playing") == "playback"
    # An explicit billing act still wins over the symptom it complains about.
    assert c._diagnostic_category("refund please, songs stop playing") == "purchases"
    # ...and status words alone still beat `account`.
    assert c._diagnostic_category("il mio account premium non si attiva") == "purchases"


@test("support_diagnostic_routing", "a capture targets the account the app declared, not the sender")
async def t_capture_uses_declared_account(_ctx: TestContext):
    from unittest.mock import AsyncMock, patch

    looked_up: list[str] = []

    async def fake_call_first(_pool, _server, _names, args, required=False):
        looked_up.append(args["query"])
        return "search_users", {"users": [{"userId": 7}]}

    state = c.SupportState("t", "i brani non partono", tenant=c._TENANTS["esound"])
    state.account_email = "sender@example.com"
    state.facts["form_account_email"] = "app-account@example.com"
    with patch.object(c, "_call_first", AsyncMock(side_effect=fake_call_first)), \
            patch.object(c, "_succeeded", lambda _r: True):
        _server, identity = await c._resolve_diagnostic_identity(None, state)
        assert identity == {"userId": 7}, identity
        assert looked_up == ["app-account@example.com"], looked_up
        # A read whose answer goes back to the customer keeps the verified
        # identity: the form is a public POST, not proof of ownership.
        looked_up.clear()
        await c._resolve_diagnostic_identity(None, state, app_account=False)
        assert looked_up == ["sender@example.com"], looked_up

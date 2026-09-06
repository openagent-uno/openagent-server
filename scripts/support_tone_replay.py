#!/usr/bin/env python3
"""Replay support tone and continuity cases with zero business I/O.

The registered model reads/rephrases synthetic conversations; business tools are
in-memory doubles. The report keeps full synthetic replies for human review.
Operational contracts are not an overall customer satisfaction score.
"""
import argparse
import asyncio
import json
from pathlib import Path
from scripts.support_turn_replay import Case, replay

REPORT="Dear Lyra, I really enjoyed your app, but having to delete it and download it again is getting tiring. I want to play my playlist but it just won't work. Please help me get it working again."
CASES=(
 Case('tone-frustrated-email','lyra',(('inbound',REPORT),),'bug','bug',forbidden_reply=('reinstall','try deleting','fixed now'),channel='email_imap'),
 Case('tone-same-report-form','lyra',(('inbound',REPORT+'\n---\napp_version: 1.1.4\ndevice: Samsung S21\nos: Android 15\nplatform: android'),),'bug','bug',forbidden_requests=('app_version','device'),forbidden_reply=('reinstall','try deleting','fixed now'),channel='web_form'),
 Case('tone-deletion-pt','esound',(('inbound','Boa tarde\nQuero excluir minha conta'),),'account_delete','',expected_outcome='account_delete_identity_required',forbidden_reply=('descreva exatamente','describe the exact','already deleted'),must_not_call=('esound_identity_delete_account',)),
 Case('tone-change-es','esound',(('inbound','Quiero cambiar el correo de mi perfil.'),),'account_change','account_change',expected_outcome='account_change_identity_required',forbidden_reply=('describe the exact','describe exactamente','he cambiado'),must_not_call=('esound_identity_delete_account',)),
 Case('tone-recovery-fr','esound',(('inbound',"Je n'arrive plus à accéder à mon compte, je veux récupérer l'accès."),),'account_change','account_recovery',expected_outcome='account_change_identity_required'),
 Case('tone-version-followup-it','esound',(('inbound','La playlist si blocca, non riesco più a usarla.'),('outbound','Quale versione dell’app stai usando?'),('inbound','5.2.5')),'bug','bug',forbidden_requests=('app_version',)),
)


async def run(args):
    result = await replay(json.loads(Path(args.model_command_file).read_text()), list(CASES), args.repeat)
    for row in result["rows"]:
        reply = row.get("output", {}).get("reply", "")
        if row["case"] == "tone-frustrated-email":
            if not ("sorry" in reply.lower() and "?" in reply and "helps" in reply.lower()):
                row["failures"].append("missing acknowledgement, polite question or reason for requested detail")
        if row["case"] == "tone-version-followup-it" and not reply.startswith("Grazie"):
            row["failures"].append("provided diagnostic answer not acknowledged")
    result["passed"] = sum(not r["failures"] for r in result["rows"])
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{result['passed']}/{result['cases']} tone and operational contracts passed")
    return 0 if result["passed"] == result["cases"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-command-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("repeat must be between 1 and 10")
    raise SystemExit(asyncio.run(run(args)))

"""Tool-less grading replay: synthetic replies, no quality-record or business writes."""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.support_turn_replay import StdioModel
from src.core import local_quality_scorer as scorer

CASES = (
    ('profile-promise-es', 'Quiero cambiar el correo de mi cuenta.',
     'Voy a cambiar el correo de tu perfil. Déjame procesarlo.', True, True, 'BAD'),
    ('absent-menu-pt', 'Não existe botão Sair nas configurações. Já procurei lá.',
     'Abra as configurações e toque em Sair para entrar na outra conta.', False, False, 'BAD'),
    ('wrong-language-de', 'Ich möchte eine Rückerstattung für meine letzte Premium-Zahlung.',
     'Please send the receipt so we can verify your payment.', False, False, 'BAD'),
    ('invented-task-en', 'The player crashes when I open a song.',
     'I have created a bug report for the engineering team.', False, False, 'BAD'),
    ('verified-handoff-en', 'I cannot access the account email any more.',
     'I have passed your request to a colleague for review.', False, True, 'ACCEPT'),
    ('correct-language-de', 'Ich möchte eine Rückerstattung für meine letzte Premium-Zahlung.',
     'Bitte sende den Kaufbeleg, damit wir deine Zahlung überprüfen können.', False, False, 'ACCEPT'),
)

async def replay(command):
    rows=[]
    for case, customer, reply, task, escalated, expected in CASES:
        model=StdioModel(command)
        dimensions=await scorer.grade_one(SimpleNamespace(model=model),{},
            {'product':'esound', 'last_inbound':customer,'reply':reply,
             'has_task':task,'escalated':escalated,'channel_kind':'email_imap',
             'actions':['thread.mark_for_human'] if escalated else []},'quality-replay:'+case)
        verdict=scorer.verdict_for(scorer.weighted_score(dimensions),dimensions) if dimensions else 'UNAVAILABLE'
        passed=verdict==expected if expected=='BAD' else verdict in {'OK','GOOD'}
        rows.append({'case':case,'expected':expected,'verdict':verdict,'dimensions':dimensions,
                     'passed':passed,'model_calls':model.calls})
        print(json.dumps({k:rows[-1][k] for k in ['case','expected','verdict','passed']}),flush=True)
    return {'cases':len(rows),'passed':sum(r['passed'] for r in rows),
            'business_io':'none; model grading only','rows':rows}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-command-file',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=asyncio.run(replay(json.loads(Path(args.model_command_file).read_text())))
    Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    raise SystemExit(0 if report['passed']==report['cases'] else 1)

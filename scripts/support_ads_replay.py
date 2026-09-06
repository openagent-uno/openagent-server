#!/usr/bin/env python3
"""Regression replies for ad objections and mixed faults; all business I/O simulated."""
import argparse
import asyncio
import json
from pathlib import Path
from scripts.support_turn_replay import Case, replay

CASES = (
    Case('mixed-ads-audio-it', 'lyra', (('inbound',
        'Mi piaceva molto la vostra app, la consigliavo agli amici. Da quando ci sono pubblicità e abbonamenti è diventata inutilizzabile: annunci durante i brani, musica che si ferma e audio accavallati. Che delusione.'),),
        'bug', 'bug', forbidden_reply=('Premium risolve', 'è normale'),
        required_reply=('malfunzionamento', 'server', 'gratis', 'Creator')),
    Case('frequency-it', 'lyra', (('inbound',
        'Le pubblicità sono troppo frequenti, non posso pagare un altro abbonamento.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained',
        required_reply=('server', 'gratis', 'invitando', 'Creator', 'Se nell’app', 'normale annuncio')),
    Case('frequency-en-review', 'esound', (('inbound',
        'There are way too many ads. An ad every two songs is unbearable.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained', channel='playstore_reviews',
        required_reply=('server', 'referral', 'video', 'if offered'), forbidden_reply=('Creator',)),
    Case('frequency-lyra-review', 'lyra', (('inbound',
        'The ads are far too frequent. I cannot afford another subscription.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained', channel='appstore_reviews',
        required_reply=('server', 'referral', 'Creator', 'video', 'if offered')),
    Case('negated-ad-fault', 'esound', (('inbound',
        'The audio is not overlapping and playback works fine. My complaint is only that there are too many ads.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained',
        required_reply=('server', 'friends'), forbidden_reply=('malfunction', 'fixed')),
    Case('pure-ad-fault', 'lyra', (('inbound',
        'After an ad finishes its overlay stays on the screen and its audio keeps playing over my music. Android 15, Samsung S21, version 5.2.4.'),),
        'bug', 'bug', forbidden_reply=('Paid Premium', 'referral program', 'Creator')),
)

async def run(args):
    result = await replay(json.loads(Path(args.model_command_file).read_text()), list(CASES), args.repeat)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{result['passed']}/{result['cases']} ad reply contracts passed")
    return 0 if result['passed'] == result['cases'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-command-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeat', type=int, default=2, choices=range(1, 11))
    raise SystemExit(asyncio.run(run(parser.parse_args())))

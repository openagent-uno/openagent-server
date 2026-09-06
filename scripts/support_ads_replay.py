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
        required_reply=('Creator',)),
    Case('frequency-it', 'lyra', (('inbound',
        'Le pubblicità sono troppo frequenti, non posso pagare un altro abbonamento.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained',
        required_reply=('30', 'Creator')),
    Case('frequency-en-review', 'esound', (('inbound',
        'There are way too many ads. An ad every two songs is unbearable.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained', channel='playstore_reviews',
        required_reply=('30',), forbidden_reply=('Creator',)),
    Case('frequency-lyra-review', 'lyra', (('inbound',
        'The ads are far too frequent. I cannot afford another subscription.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained', channel='appstore_reviews',
        required_reply=('Creator', '30')),
    Case('negated-ad-fault', 'esound', (('inbound',
        'The audio is not overlapping and playback works fine. My complaint is only that there are too many ads.'),),
        'premium', 'ads_feedback', expected_outcome='ads_policy_explained',
        required_reply=('30',), forbidden_reply=('malfunction', 'fixed')),
    Case('pure-ad-fault', 'lyra', (('inbound',
        'After an ad finishes its overlay stays on the screen and its audio keeps playing over my music. Android 15, Samsung S21, version 5.2.4.'),),
        'bug', 'bug', forbidden_reply=('Paid Premium', 'referral program', 'Creator')),
)

async def run(args):
    result = await replay(json.loads(Path(args.model_command_file).read_text()), list(CASES), args.repeat)
    for row in result['rows']:
        out=row.get('output',{});reply=out.get('reply','').casefold()
        if not reply or out.get('facts',{}).get('reply_source')!='model:human_voice_verified':
            row['failures'].append('no reviewed contextual reply delivered')
        if row['case'] != 'pure-ad-fault':
            for label, alternatives in (
                ('server costs', ('server','infrastruttur','running costs')),
                ('referral', ('referr','invit','friend')),
                ('one accumulated hour', ('60','un’ora',"un'ora",'one hour','an hour','a full hour')),
                ('settings', ('settaggi','impostazioni','settings')),
            ):
                if not any(word in reply for word in alternatives):
                    row['failures'].append('missing explanation: '+label)
    result['passed']=sum(not row['failures'] for row in result['rows'])
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{result['passed']}/{result['cases']} ad reply contracts passed")
    return 0 if result['passed'] == result['cases'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-command-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeat', type=int, default=2, choices=range(1, 11))
    raise SystemExit(asyncio.run(run(parser.parse_args())))

"""Regressions from post-0.20.55 customer conversations (sanitized)."""
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from scripts.tests._framework import test
from scripts.tests.test_support_voice import state, VoiceModel, OK
from scripts.tests.test_local_support_controller import _Doubles, _Toolkit, _Pool
from src.core import local_support_controller as c, support_voice as v
from src.core import support_guidance as g, support_delivery_receipts as receipts


@test('support_sept7', 'MCP document and attachment envelopes preserve evidence, never errors')
async def envelopes(_):
    doc='In eSound open Settings, then About, then Version. Read the installed app version number there.'
    hit={'items':[{'excerpt':doc}]}
    for wrapped in [hit, {'structuredContent':hit}, {'content':[{'type':'text','text':json.dumps(hit)}]},
                    [{'type':'text','text':hit}]]:
        assert g.excerpts(wrapped)==[doc],wrapped
    assert not g.excerpts({'isError':True,'structuredContent':hit})
    assert c._attachment_text({'content':[{'type':'text','text':json.dumps({'text':'A thumbs up reaction'})}]})=='A thumbs up reaction'
    s=state();s.outcome='guidance_answer';s.facts['guidance_documents']=hit
    packet=v.packet(s,c._fallback_reply(s),1800)
    assert v.guidance_supported({'product_steps_present':True,'source_quotes':['In eSound open Settings,\nthen About, then Version.']},packet)
    assert not v.guidance_supported({'product_steps_present':True,'source_quotes':['In eSound open Settings, then Developer menu.']},packet)


@test('support_sept7', 'reclaimed escalation obtains and verifies ownership before retry, once per turn')
async def handoff(_):
    s=state();s.human_reason='The reply was held; answer the outstanding pause/resume request.'
    calls=[];linked=''
    async def mark(**kw):
        calls.append('mark')
        return {'ok':True} if linked else {'isError':True,'content':[{'type':'text','text':'409 human-lane-timeout: link a task first'}]}
    async def own(**kw):
        nonlocal linked
        calls.append('own');linked='case-123';return {'linked':True,'external_task_id':linked}
    async def get(**kw):
        calls.append('get');return {'external_task_id':linked}
    pool=_Pool({'replio':_Toolkit({'threads_mark_for_human':mark,'thread_ensure_support_task':own,'threads_get':get,
                                 'threads_tags_add':AsyncMock(return_value={'ok':True})})})
    with patch.dict(os.environ,{c._WRITES_ENV:'1'}):
        assert await c._queue_for_human(pool,s)
        assert await c._queue_for_human(pool,s)
    assert calls==['mark','own','get','mark'],calls
    assert s.facts['human_owner_verified'] and s.linked_task_id=='case-123'


@test('support_sept7', 'unverified support task cannot authorize a handoff or blind create retry')
async def missing_owner(_):
    s=state();s.human_reason='Unresolved support case'
    mark=AsyncMock(return_value={'isError':True,'text':'human-lane-timeout'})
    own=AsyncMock(return_value={'external_task_id':'claimed'})
    pool=_Pool({'replio':_Toolkit({'threads_mark_for_human':mark,'thread_ensure_support_task':own,
        'threads_get':AsyncMock(return_value={'external_task_id':None}),'threads_tags_add':AsyncMock(return_value={'ok':True})})})
    with patch.dict(os.environ,{c._WRITES_ENV:'1'}):
        assert not await c._queue_for_human(pool,s)
        assert not await c._queue_for_human(pool,s)
    assert own.await_count==mark.await_count==1
    assert s.facts['human_handoff_error']=='owner_not_verified'


@test('support_sept7', 'repeated unreadable attachments do not trigger another transcription demand')
async def repeated_attachment(_):
    s=state();s.prior_support_replies=['Puoi incollare qui il testo direttamente nel messaggio?']
    s.facts['attachment_readable']=False
    await c._route_attachment(_Doubles().pool(),s)
    assert s.decision=='human' and s.outcome=='attachment_partial_review'
    assert 'resend or transcribe' in s.human_reason


@test('support_sept7', 'long success envelope retains actual blocked delivery and owner failure as valid JSON')
async def retained(_):
    from src.core.event_dispatcher import _retained_output
    data={'controller':'esound-local-v1','thread_id':'synthetic','intent':'bug','outcome':'bug_needs_evidence',
          'reply':'Friendly but incomplete answer.','actions':[{'kind':'customer_reply','success':False,
          'receipt':{'sent':False,'blocked':True,'category':'ignores_question','reason':'x'*12000}}],
          'facts':{'delivery_state':'blocked','human_handoff_confirmed':False,'human_handoff_error':'owner_not_verified'}}
    text=_retained_output(json.dumps(data));out=json.loads(text)
    assert len(text)<=4000 and out['reply']==data['reply']
    assert out['delivery']['state']=='blocked' and out['delivery']['category']=='ignores_question'
    assert out['facts']['human_handoff_confirmed'] is False
    assert _retained_output(text)==text


@test('support_sept7', 'direct behaviour request survives into writer and rejects an otherwise friendly omission')
async def ignored_request(_):
    s=state();s.customer_message='Could you pause the song during the ad and resume afterward?'
    s.instructions=['Address the requested pause/resume behaviour; no change has been made.']
    first='Sorry this is frustrating. Could you send the app version?'
    final='I understand why you want playback to pause during the ad and resume afterward. I have not changed that behaviour on your device. Could you share your app version so we can investigate the overlap?'
    m=VoiceModel([first,final],[{**OK,'answers_customer':False,'findings':['Answer the requested pause and resume behaviour first.']},OK])
    with patch.dict(os.environ,{v.ENV:'1'}):
        result=await c._compose_human_reply(SimpleNamespace(model=m,_mcp=_Doubles().pool()),{},s,'test')
    assert result==final and s.facts['human_voice_attempts']==2
    packet=json.loads(m.calls[0]['messages'][0]['content'])
    assert packet['task_instructions']==s.instructions

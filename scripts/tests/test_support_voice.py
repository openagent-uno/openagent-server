"""Production-default voice, failure handling and delivery integrity without I/O."""
import asyncio
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
from scripts.tests._framework import test
from scripts.tests.test_local_support_controller import _Doubles
from src.core import local_support_controller as c, support_voice as v

OK={k:True for k in ('facts_supported','required_content_preserved','answers_customer','humane','language_correct')}

class VoiceModel:
    def __init__(self, replies, verdicts=None):
        self.replies=iter(replies);self.verdicts=iter(verdicts or [OK]*4);self.calls=[]
    async def generate(self, **kw):
        from src.core.tool_scope import current_tool_allowlist
        from src.core.execution_profile import strict_local_only_active
        assert current_tool_allowlist()==frozenset() and strict_local_only_active()
        self.calls.append(kw)
        if kw['system']==v.REVIEW_SYSTEM:
            packet=json.loads(kw['messages'][0]['content'])
            verdict=next(self.verdicts)
            verdict={**verdict,'product_steps_present':False,'source_quotes':[], 'coverage':[{'point':i,'quote':packet['proposed_reply']} for i in range(len(packet['required_points']))]}
            return SimpleNamespace(content=json.dumps(verdict))
        value=next(self.replies)
        if isinstance(value,Exception):raise value
        return SimpleNamespace(content=json.dumps({'language':'en','reply':value}))


def state():
    s=c.SupportState('synthetic','I recommended your app to my friends, but my playlists fail.',
        intent='bug',decision='ask_information',outcome='bug_needs_evidence',
        facts={'language':'en','missing_evidence':['app version']})
    s.recent_exchange=[{'from':'customer','text':s.customer_message}]
    return s


@test('support_voice','every guarded outcome uses the writer and preserves conversation context')
async def all_routes(_):
    with patch.dict(os.environ,{v.ENV:'1'}):
        for outcome,intent,decision in [
            ('bug_needs_evidence','bug','ask_information'),('ads_policy_explained','premium','self_help'),
            ('premium_active','premium','self_help'),('account_delete_confirmation_required','account_delete','ask_information'),
            ('account_change_identity_required','account_change','ask_information'),('general_needs_detail','general','ask_information'),
            ('account_deleted','account_delete','self_help')]:
            s=state();s.outcome=outcome;s.intent=intent;s.decision=decision
            brief=c._fallback_reply(s);reply='Thanks for explaining. '+brief
            model=VoiceModel([reply]);d=_Doubles()
            actual=await c._compose_local(SimpleNamespace(model=model,_mcp=d.pool()),{},s,'unit')
            assert actual==reply and s.facts['reply_source']=='model:human_voice_verified'
            assert s.facts['human_voice_sha256']==hashlib.sha256(reply.encode()).hexdigest()
            packet=json.loads(model.calls[0]['messages'][0]['content'])
            assert packet['recent_exchange']==s.recent_exchange and packet['operational_brief']==brief


@test('support_voice','a factual or cold reply is repaired instead of replaced by a canned guard answer')
async def repair(_):
    with patch.dict(os.environ,{v.ENV:'1'}):
        s=state();reply="I'm sorry playlists are failing. Could you share the app version so we can investigate?"
        model=VoiceModel(['Send the app version.',reply],[{**OK,'humane':False,'findings':['Ask politely and explain why.']},OK])
        out=await c._compose_local(SimpleNamespace(model=model,_mcp=_Doubles().pool()),{},s,'unit')
        assert out==reply and s.facts['human_voice_attempts']==2
        assert 'Ask politely' in json.loads(model.calls[2]['messages'][0]['content'])['reviewer_findings']


@test('support_voice','provider failure holds work without sending or closing the inquiry')
async def failed(_):
    with patch.dict(os.environ,{v.ENV:'1',c._WRITES_ENV:'1'}):
        s=state();d=_Doubles();model=VoiceModel([RuntimeError('provider unavailable')])
        out=await c._compose_local(SimpleNamespace(model=model,_mcp=d.pool()),{},s,'unit')
        await c._apply_lifecycle(d.pool(),s,out)
        assert out=='' and 'replio_threads_respond' not in d.names
        assert 'replio_threads_mark_for_human' in d.names
        assert not any(x.get('patch',{}).get('status')=='closed' for x in d.args_for('replio_threads_patch'))


@test('support_voice','invented amounts never reach the reviewer or customer')
async def fabricated(_):
    with patch.dict(os.environ,{v.ENV:'1'}):
        s=state();model=VoiceModel(['We refunded 999 euros.']*3);d=_Doubles()
        out=await c._compose_local(SimpleNamespace(model=model,_mcp=d.pool()),{},s,'unit')
        assert out=='' and len(model.calls)==3 and 'replio_threads_respond' not in d.names


@test('support_voice','text changed after review cannot be delivered as a fallback')
async def delivery_seal(_):
    with patch.dict(os.environ,{v.ENV:'1',c._WRITES_ENV:'1'}):
        s=state();d=_Doubles();s.facts['human_voice_sha256']=hashlib.sha256(b'approved').hexdigest()
        await c._apply_lifecycle(d.pool(),s,'Send app version.')
        assert 'replio_threads_respond' not in d.names and 'replio_threads_mark_for_human' in d.names


@test('support_voice','a guard finding re-enters the writer instead of returning its canned sentence')
async def final_guard(_):
    with patch.dict(os.environ,{v.ENV:'1'}):
        s=state();reply='Thanks for reporting this. Could you share the app version?'
        m=VoiceModel([reply]);out=await c._validate_final_reply(SimpleNamespace(model=m,_mcp=_Doubles().pool()),{},s,"I'm a human support agent.",'unit')
        assert out==reply and s.facts['reply_source']=='model:human_voice_verified'


@test('support_voice','incomplete or string-valued review verdicts fail closed')
async def verdicts(_):
    assert v.accepted(OK)
    assert not v.accepted({**OK,'humane':'true'})
    assert not v.accepted({'humane':True})
    assert not v.covered({**OK,'coverage':[{'point':0,'quote':'invented quote'}]},['required fact'],'actual reply')
    assert not v.covered({**OK,'coverage':[]},['required fact'],'actual reply')
    assert v.unsupported_duration('Remove ads permanently through referrals.', 'Earn Premium through referrals.')
    assert not v.unsupported_duration('Watch videos for temporary ad-free time.', 'Earn 30 minutes.')
    assert not v.unsupported_duration('This is a lifetime benefit.', 'Verified lifetime benefit.')


@test('support_voice','reviewed prose reaches the delivery tool unchanged and respects real channel caps')
async def delivery(_):
    with patch.dict(os.environ,{v.ENV:'1',c._WRITES_ENV:'1'}):
        s=state();d=_Doubles();reply='Thanks for reporting this. Could you share the app version?'
        out=await c._compose_local(SimpleNamespace(model=VoiceModel([reply]),_mcp=d.pool()),{},s,'unit')
        await c._apply_lifecycle(d.pool(),s,out)
        assert d.args_for('replio_threads_respond')[0]['body_text']==reply
        assert c._reply_cap('playstore_reviews')==350
        assert c._reply_cap('appstore_reviews')==900
        assert c._reply_cap('email_imap')==1800


@test('support_voice','short reviews get a bounded extra edit without truncating approved meaning')
async def short_review(_):
    with patch.dict(os.environ,{v.ENV:'1'}):
        s=state();s.channel='playstore_reviews';d=_Doubles()
        reply='Sorry playlists are failing. Could you share the app version so we can investigate?'
        model=VoiceModel(['x'*420,'x'*354,reply])
        out=await c._compose_local(SimpleNamespace(model=model,_mcp=d.pool()),{},s,'unit')
        assert out==reply and s.facts['human_voice_attempts']==3
        packet=json.loads(model.calls[2]['messages'][0]['content'])
        assert '354 characters' in packet['reviewer_findings']
        assert packet['previous_attempt']=='x'*354


@test('support_voice','guidance receives retrieved references instead of the generic fallback question')
async def guidance_context(_):
    s=state();s.outcome='guidance_answer';s.intent='guidance_question'
    s.policy_notes={'product.md':'Verified product guidance belongs here.'}
    s.facts['already_known_from_form']={'app_version':'5.2.4'}
    packet=v.packet(s,c._fallback_reply(s),1800)
    assert 'What issue' not in packet['operational_brief']
    assert packet['required_points']==[]
    assert 'Verified product guidance' in str(packet['supporting_material'])
    assert packet['already_provided']['app_version']=='5.2.4'
    assert not v.guidance_supported({'product_steps_present':True,'source_quotes':['invented source']},packet)
    assert not v.guidance_supported({'product_steps_present':True,'source_quotes':[]},packet)
    assert v.guidance_supported({'product_steps_present':True,'source_quotes':['Verified product guidance belongs here.']},packet)
    from src.core.support_turn import requested_fields
    assert requested_fields('Could you share your device model and operating system version?')=={'device','os'}
    assert not requested_fields("Could you send a recording? That helps me see what is going wrong on your device.")
    assert requested_fields('What app version are you using')=={'app_version'}
    assert not requested_fields("I can't verify those steps for your device, so I'd rather not send you down the wrong path.")
    assert not c._introduces_claim('Your phone runs Android 15.', 'Reported Android 15, Samsung S21', s)
    assert c._introduces_claim('Your phone runs Android 16.', 'Reported Android 15, Samsung S21', s)
    assert 'app_version' not in requested_fields('Qual dispositivo e versão do sistema operacional você usa?')
    assert 'app_version' in requested_fields('Qual versão do aplicativo você usa no Android?')
    assert 'app_version' not in requested_fields('Per approfondire il blocco nella versione 5.2.5, potresti indicarmi il dispositivo e la versione del sistema operativo?')


@test('support_voice','one deadline bounds the whole writing cycle including subsequent guard retries')
async def voice_budget(_):
    class SlowModel:
        async def generate(self, **kwargs):
            await asyncio.sleep(.05)
            raise AssertionError('deadline should expire first')
    with patch.dict(os.environ,{v.ENV:'1'}):
        s=state();s.voice_deadline=asyncio.get_running_loop().time()+.005
        deadline=s.voice_deadline;d=_Doubles();agent=SimpleNamespace(model=SlowModel(),_mcp=d.pool())
        assert await c._compose_local(agent,{},s,'unit')==''
        assert await c._compose_local(agent,{},s,'unit-retry')==''
        assert s.voice_deadline==deadline and 'replio_threads_respond' not in d.names

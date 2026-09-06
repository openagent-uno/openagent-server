"""Regression contracts from the September support audit; no business I/O."""
from types import SimpleNamespace
import json
import os
from unittest.mock import patch
from ._framework import test
from .test_local_support_controller import _Doubles, _Model
from src.core import local_support_controller as c, reply_guard as g, support_turn

@test("support_sept6", "account recovery and conditional refunds preserve the main request")
async def routing(_ctx):
    assert c._intent("Quiero recuperar mi cuenta") == "account_change"
    assert c._intent("L'app si blocca quando apro una canzone. Se no sarò costretto a chiedere il rimborso") == "bug"
    assert c._intent("I want a refund for my payment") == "refund"
    assert c._intent("I see two charges; refund the duplicate") == "duplicate_charge"

@test("support_sept6", "account promise cannot escape by being in the future")
async def authority(_ctx):
    for claim in ["Voy a cambiar el correo de tu perfil.", "I will change your email.", "Ho cambiato il profilo.", "Vou alterar o email da conta."]:
        assert g.claims_profile_change(claim), claim
    for instruction in ["You can change your email in Settings.", "A human can review your request.", "I will ask a colleague to review the account."]:
        assert not g.claims_profile_change(instruction), instruction
    s=c.SupportState("test", "Change my email", intent="account_change")
    assert not c._model_may_compose(s)

@test("support_sept6", "an untranslatable reply is held without sending English")
async def failed_translation(_ctx):
    class Unavailable:
        async def generate(self, **kw): raise TimeoutError()
    d=_Doubles()
    s=c.SupportState("test", "Bitte helfen Sie mir", outcome="account_change_identity_required",intent="account_change", facts={"language":"de"})
    with patch.dict(os.environ, {c._WRITES_ENV:"1"}):
        reply=await c._fallback_in_language(SimpleNamespace(_mcp=d.pool(),model=Unavailable()),{},s,"test","test")
    assert reply == ""
    assert "replio_threads_respond" not in d.names
    assert "replio_threads_mark_for_human" in d.names

@test("support_sept6", "translation cannot invent an account operation")
async def malicious_translation(_ctx):
    class Wrong:
        async def generate(self, **kw):return SimpleNamespace(content=json.dumps({"reply":"Voy a cambiar el correo de tu perfil."}))
    d=_Doubles();s=c.SupportState("test","Quiero cambiar mi correo",intent="account_change",outcome="account_change_identity_required",facts={"language":"es"})
    with patch.dict(os.environ,{c._WRITES_ENV:"1"}):
        reply=await c._fallback_in_language(SimpleNamespace(_mcp=d.pool(),model=Wrong()),{},s,"test","authority")
    assert reply == "" and "replio_threads_respond" not in d.names

@test("support_sept6", "notices and held cases never enter a customer reply")
async def notices(_ctx):
    for thread,message in [
        ({"tags":["automated-notice"]},"Your submission is ready"),
        ({"messages":[{"direction":"inbound","author_handle":"noreply@email.apple.com","body_text":"Ready"}]},"Ready"),
        ({"tags":["support-review-required"]},"My player is broken"),
    ]:
        d=_Doubles(thread=thread)
        with patch.dict(os.environ,{c._WRITES_ENV:"1"}):
            out=json.loads((await c.run(agent=SimpleNamespace(_mcp=d.pool(),model=_Model()),event={"slug":"replio-thread"},payload={"payload":{"thread_id":"test","message":{"body_text":message}}},session_id="test",delivery_id="test")).text)
        assert out["reply"] == "" and "replio_threads_respond" not in d.names

@test("support_sept6", "business enquiries are assigned without a support questionnaire")
async def business(_ctx):
    d=_Doubles()
    with patch.dict(os.environ,{c._WRITES_ENV:"1"}):
        out=json.loads((await c.run(agent=SimpleNamespace(_mcp=d.pool(),model=_Model()),event={"slug":"replio-thread"},payload={"payload":{"thread_id":"test","message":{"body_text":"We propose a business partnership with your company."}}},session_id="test",delivery_id="test")).text)
    assert "replio_threads_mark_for_human" in d.names
    assert out["reply"] == "" and "replio_threads_respond" not in d.names

@test("support_sept6", "reader can correct a money topic without authorizing a mutation")
async def reader(_ctx):
    class Model:
        async def generate(self,**kw):return SimpleNamespace(content=json.dumps({"kind":"account_recovery","evidence":"recuperar mi cuenta"}))
    s=c.SupportState("test","Quiero recuperar mi cuenta",intent="refund",facts={"intent_source":"semantic","money_execution_requires_human":True})
    await c._read_reported_turn(SimpleNamespace(model=Model()),{},s,"test")
    assert s.intent == "account_change" and s.facts["money_execution_requires_human"]

@test("support_sept6", "a quoted correction holds unavailable UI instructions for a real review")
async def unavailable_instruction(_ctx):
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({
                "kind":"bug", "evidence":"The Log Out button does not exist",
                "reported":{"unavailable_instruction":"The Log Out button does not exist"}}))
    s=c.SupportState("test","The Log Out button does not exist",intent="bug")
    s.prior_support_replies=["Tap Log Out in Settings."]
    await c._read_reported_turn(SimpleNamespace(model=Model()),{},s,"test")
    assert s.intent == "guidance_question" and s.facts['unavailable_instruction']
    s.outcome='guidance_unavailable_human'
    assert c._fallback_reply(s)==''  # No invented handoff if queuing failed.
    s.facts['human_handoff_confirmed']=True
    assert 'colleague' in c._fallback_reply(s) and 'Tap Log Out' not in c._fallback_reply(s)
    bad=support_turn.read_reported_turn({'kind':'bug','evidence':'button','reported':{'unavailable_instruction':'Imaginary customer correction'}},s.customer_message)
    assert bad and 'unavailable_instruction' not in bad.reported

@test("support_sept6", "completed Apple notices are recognized before semantic acknowledgement")
async def apple_notice(_ctx):
    assert c._is_machine_mail('We have completed our review of your submission.')
    assert not c._is_machine_mail('Can you review my payment?')

@test("support_sept6", "the grader sees action evidence and the reviewer's declared language")
async def grader_evidence(_ctx):
    from src.core import local_quality_scorer as scorer
    class Model:
        packet=None
        async def generate(self, **kw):
            self.packet=json.loads(kw['messages'][0]['content'])
            return SimpleNamespace(content=json.dumps({key:1 for key in scorer._DIMENSIONS}))
    model=Model()
    result=await scorer.grade_one(SimpleNamespace(model=model),{},
        {'product':'esound','last_inbound':'Too many ads\n---\nreviewer_language: pt','reply':'Entendo.',
         'actions':['thread.mark_for_human']},'test')
    assert result and model.packet['reviewer_language']=='pt'
    assert model.packet['actions']==['thread.mark_for_human'] and model.packet['product']=='esound'

@test("support_sept6", "a model's perfect scores cannot authorize a profile promise")
async def grader_profile_promise(_ctx):
    from src.core import local_quality_scorer as scorer
    class Model:
        async def generate(self, **kw):
            return SimpleNamespace(content=json.dumps({key:1 for key in scorer._DIMENSIONS}))
    result=await scorer.grade_one(SimpleNamespace(model=Model()),{},
        {'product':'esound','last_inbound':'Change my email','reply':'Voy a cambiar el correo de tu perfil.',
         'has_task':True,'escalated':True},'test')
    assert result['grounding']==0 and scorer.verdict_for(scorer.weighted_score(result),result)=='BAD'

"""Model-backed resolution replay with the real Replio close contracts.

All business tools mutate only an in-memory case. The sole external process
is the tool-less registered model adapter. No customer messages are sent.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts.support_turn_replay import StdioModel
from scripts.tests.test_local_support_controller import _Doubles, _Toolkit
from src.core import local_support_controller as c
from src.core.dry_run import dry_run_scope

CASES = [
    {"name":"documented-android-auto-es", "product":"lyra", "turns":[
        ("inbound","Instalé Lyra desde el APK de la web. No aparece en Android Auto. ¿Puede ser por cómo la instalé?\n---\napp_version: 1.4.11\ndevice: Redmi Note 11\nos: Android 15\nplatform: android")],
     "docs":"Android Auto only lists apps installed from the Google Play Store. Installed from lyramusic.app/download (com.musicapp.lyra.direct): the restriction still applies, and it is the explanation. Installing from the Play listing is now a real option to offer. The sideload restriction is a platform rule, not a fault in the app and not something a subscription unlocks.",
     "outcome":"guidance_verified", "human":False,"status":"open","contains":"Google Play","language":"es","forbidden":["send a log","gravação","reproduce again"]},
    {"name":"pending-payment-order", "turns":[
        ("inbound","My Google Play payment has been processing for a week. I need Premium.\n---\naccount_email: customer@example.test\naccount_user_id: aaaaaaaaaaaaaaaaaaaaaaaa"),
        ("outbound","Please send the Google Play order number."),
        ("inbound","GPA.0000-1111-2222-33333"),
        ("inbound","How can I fix the pending payment? I need Premium.")],
     "outcome":"billing_unverified_human", "human":True, "status":"open", "contains":"order", "forbidden":["please send the order","send your receipt","refund eligibility"]},
    {"name":"requested-playlist-link", "task":"task-synthetic", "turns":[
        ("inbound","My playlist import fails."),
        ("outbound","Please send the original playlist link so we can reproduce it."),
        ("inbound","Should I send my playlist link?")],
     "outcome":"pending_playlist_link_answer", "human":False,"status":"open","contains":"Yes", "forbidden":["no need to send","sending it again won't help"]},
    {"name":"honest-identity", "turns":[("inbound","Are you a robot?")],
     "outcome":"bot_identity_answer", "human":False,"status":"open", "contains":"automated", "forbidden":["not a robot","a person helping"]},
    {"name":"courtesy-keeps-obligation", "tags":["awaiting-user","needs-human"], "turns":[
        ("inbound","Playback still fails."), ("outbound","A colleague will investigate the issue."), ("inbound","Thank you")],
     "human":True,"status":"open","silent":True,"hold":True},
    {"name":"confirmed-resolution-closes", "tags":["awaiting-user"], "turns":[
        ("inbound","The phone vibrates while playing songs."), ("outbound","Which app version do you use?"),
        ("inbound","Hello. It happens in every song after last update. I switched off the aptics and the problem solved.")],
     "outcome":"resolved_confirmation","human":False,"status":"closed","silent":True},
    {"name":"verified-task-status", "task":"task-synthetic", "turns":[
        ("inbound","The app crashes on startup."), ("outbound","Your report is attached to the task."),
        ("inbound","Is there an update on the bug I reported? Is it fixed?")],
     "outcome":"status_task_verified","human":False,"status":"open", "contains":"not marked complete","forbidden":["please send","is fixed","install the latest"]},
]


class CaseWorld(_Doubles):
    def __init__(self, case, guard):
        self.case=case; self.guard=guard
        super().__init__(thread={"product":case.get("product","esound"),"status":"open","tags":list(case.get("tags",[])),
            "external_task_id":case.get("task"), "messages":[{"direction":d,"body_text":b} for d,b in case["turns"]]},
            customer={"ok":True,"appUserId":"aaaaaaaaaaaaaaaaaaaaaaaa","isPremium":False,"subscriptions":[]})
        self.sent=[];self.human=False

    def pool(self):
        pool=super().pool()
        async def respond(thread_id,body_text,**kw):
            self._log("replio_threads_respond",body_text=body_text)
            self.sent.append(body_text);self._thread["messages"].append({"direction":"outbound","body_text":body_text})
            return {"ok":True,"sent":True,"simulated":True}
        async def tags(thread_id,tags):
            self._log("replio_threads_tags_add",tags=tags)
            self._thread["tags"]=list(set(self._thread["tags"])|set(tags))
            return {"ok":True,"simulated":True}
        async def patch_thread(thread_id,patch):
            self._log("replio_threads_patch",patch=patch)
            if patch.get("status") in {"closed","archived"}:
                self.guard.assert_no_silent_close(new_status=patch["status"],current_status=self._thread["status"],tags=self._thread["tags"],messages=self._thread["messages"])
                self.guard.assert_thread_close_answers_the_customer(new_status=patch["status"],tags=self._thread["tags"],messages=self._thread["messages"])
            self._thread.update(patch)
            return {"ok":True,"simulated":True}
        async def human(thread_id,reason):
            self._log("replio_threads_mark_for_human",reason=reason);self.human=True
            return {"ok":True,"simulated":True}
        async def docs_search(query,product,limit=4):
            self._log("replio_docs_search",product=product)
            return {"items":[{"excerpt":self.case["docs"]}]} if self.case.get("docs") else {"items":[]}
        async def get_task(task_id):
            self._log("clickup_get_task",task_id=task_id)
            return {"id":task_id,"status":{"status":"in progress"}}
        pool._toolkit_by_name["replio"].functions.update(_Toolkit({
            "replio_docs_search":docs_search,"replio_threads_respond":respond,"replio_threads_tags_add":tags,
            "replio_threads_patch":patch_thread,"replio_threads_mark_for_human":human}).functions)
        pool._toolkit_by_name["clickup"].functions.update(_Toolkit({"clickup_get_task":get_task}).functions)
        return pool


async def replay(args):
    spec=importlib.util.spec_from_file_location("replio_close_guard",Path(args.replio_source)/"backend/close_guard.py")
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    command=json.loads(Path(args.model_command_file).read_text())
    os.environ.update({"OPENAGENT_FORCE_DRY_RUN":"1","OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES":"1", "OPENAGENT_SUPPORT_TURN_READER":"1","OPENAGENT_SUPPORT_SEMANTIC_ROUTING":"0"})
    rows=[]
    for case in CASES:
        if args.case and case["name"] not in args.case:
            continue
        for iteration in range(args.repeat):
            world=CaseWorld(case,guard);model=StdioModel(command);failures=[]
            async def turn():
                with dry_run_scope(True):
                    return json.loads((await c.run(agent=SimpleNamespace(_mcp=world.pool(),model=model),event={"slug":"replio-thread"},payload={"payload":{"thread_id":"synthetic-case","product":case.get("product","esound"),"channel_kind":"email_imap","message":{"body_text":case["turns"][-1][1]}}},session_id="resolution-replay",delivery_id="synthetic")).text)
            try:
                out=await turn();reply=out["reply"]
                if case.get("outcome") and out["outcome"]!=case["outcome"]:failures.append("outcome: "+out["outcome"])
                if world._thread["status"]!=case["status"]:failures.append("wrong persisted status")
                if world.human!=case["human"]:failures.append("unjustified or missing handoff")
                if case.get("language") and out.get("language")!=case["language"]:failures.append("wrong customer language")
                if case.get("silent") and world.sent:failures.append("unnecessary reply")
                if case.get("contains","").casefold() not in reply.casefold():failures.append("missing useful next step")
                if any(text.casefold() in reply.casefold() for text in case.get("forbidden",[])):failures.append("repeated request or false claim")
                if case.get("hold"):
                    count=len(world.calls);again=await turn()
                    if again["outcome"]!="support_review_no_reply":failures.append("rejected disposition retried")
                    if any(n in {"replio_threads_patch","replio_threads_respond"} for n,_ in world.calls[count:]):failures.append("held case changed")
                rows.append({"case":case["name"],"iteration":iteration+1,"failures":failures,"output":out,"persisted_status":world._thread["status"],"handoff":world.human,"model_calls":model.calls})
            except Exception as exc:
                rows.append({"case":case["name"],"iteration":iteration+1,"failures":[type(exc).__name__+": "+str(exc)[:250]]})
            print(json.dumps({k:rows[-1][k] for k in ["case","iteration","failures"]}),flush=True)
    return {"cases":len(rows),"passed":sum(not r["failures"] for r in rows),"business_io":"in-memory tools; actual Replio close guards; tool-less model adapter","rows":rows}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-command-file",required=True)
    parser.add_argument("--replio-source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--repeat",type=int,default=2)
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    result=asyncio.run(replay(args));Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    raise SystemExit(0 if result["passed"]==result["cases"] else 1)

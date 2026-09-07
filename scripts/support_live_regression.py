"""Sanitized real multi-turn failures; actual reader/writer, simulated operations.
Use --docs-command-file for the real Replio search implementation over the revised
corpus and --model-command-file for a registered tool-less model adapter.
"""
import argparse,asyncio,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from scripts.support_turn_replay import StdioModel
from scripts.tests.test_local_support_controller import _Doubles
from src.core import local_support_controller as c
from src.core.dry_run import dry_run_scope

CASES=[
 {'id':'esound-find-version-ios','product':'esound','turns':[
 ('inbound','No puedo escuchar mis canciones, aparece un error.'),('outbound','¿Qué versión de la app tienes?'),
 ('inbound','¿Cómo puedo ver qué versión tengo?'),('inbound','iOS')], 'required':['versi','config|ajustes','acerca|informaci']},
 {'id':'lyra-integrations-google','product':'lyra','turns':[
 ('inbound','What does integrations do? I tried to add YouTube there and it just gives me a Google page.\n---\napp_version: 1.4.11\ndevice: Samsung A15\nos: Android 16')], 'required':['google','youtube'], 'forbidden':['full catalog','full catalogue']},
 {'id':'esound-login-help','product':'esound','turns':[('inbound','Quiero iniciar sesión')], 'required':['sesi|entrar|acced']},
 {'id':'lyra-ad-pause-request','product':'lyra','turns':[('inbound','When ads play, the song keeps going simultaneously with the ad. Could you stop the song when an ad starts and resume it once it finishes?')], 'required':['paus|stop','resum|again|after'], 'forbidden':['referral','Creator']},
 {'id':'lyra-ads-and-malfunction','product':'lyra','turns':[('inbound','Ho consigliato Lyra a tutti i miei amici. Ora ci sono troppe pubblicità, e la musica si sovrappone alla pubblicità: potete fermarla durante il video e farla ripartire alla fine?')], 'required':['30','server','pubblicit','ripart|ripren|paus'], 'forbidden':['sempre senza pubblicità']},
 {'id':'esound-unreadable-repeat','product':'esound','turns':[('inbound','Non riesco a installare eSound su iPhone.'),('outbound','Puoi incollare qui il testo del messaggio?'),('inbound','[1 attachment(s): text]')], 'attachment':{'content':[{'type':'text','text':json.dumps({'text':'Anteprima link: https://esound.app/altstore'})}]}, 'forbidden':['incoll','trascr','riscrivi','qual è il problema'], 'required':['altstore','safari|account apple|regione|paese|requisit|compatibil|idone']},
 {'id':'esound-reclaimed-hold','product':'esound','turns':[('inbound','I would like to change the email associated with my account.')], 'reclaimed':True, 'block':True, 'required':[]},
]
async def run(args):
 os.environ.update(OPENAGENT_FORCE_DRY_RUN='1',OPENAGENT_SUPPORT_HUMAN_VOICE='1',OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES='1',OPENAGENT_SUPPORT_TURN_READER='1',OPENAGENT_SUPPORT_SEMANTIC_ROUTING='0',OPENAGENT_SUPPORT_VOICE_MODEL=args.voice_model,OPENAGENT_SUPPORT_VISION_MODEL=args.voice_model)
 rows=[];command=json.loads(Path(args.model_command_file).read_text());docs_command=json.loads(Path(args.docs_command_file).read_text())
 for case in CASES:
  if args.case and case['id'] not in args.case:continue
  for iteration in range(args.repeat):
   messages=[{'direction':d,'body_text':t,'external_message_id':f'fixture-{i}'} for i,(d,t) in enumerate(case['turns'])]
   if case.get('attachment'):messages[-1]['attachments']=[{'name':'preview.txt'}]
   thread={'product':case['product'],'messages':messages,'tags':['human-lane-timeout'] if case.get('reclaimed') else []}
   d=_Doubles(thread=thread,attachment=case.get('attachment'));pool=d.pool();tk=pool.toolkit_by_name('replio')
   async def docs(**kw):
    d._log('replio_docs_search',**kw)
    p=await asyncio.create_subprocess_exec(*docs_command,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    out,err=await p.communicate(json.dumps(kw).encode())
    if p.returncode:raise RuntimeError(err.decode()[-200:])
    return json.loads(out)
   tk.functions['replio_docs_search']=SimpleNamespace(entrypoint=docs,parameters={'properties':{}})
   if case.get('reclaimed'):
    async def mark(**kw):
     d._log('replio_threads_mark_for_human',**kw)
     return {'ok':True,'simulated':True} if d._links else {'isError':True,'text':'409 human-lane-timeout: link a task first'}
    async def own(**kw):
     d._log('replio_thread_ensure_support_task',**kw);d._links[kw['thread_id']]='support-case-sim'
     return {'external_task_id':'support-case-sim','linked':True,'simulated':True}
    tk.functions['replio_threads_mark_for_human']=SimpleNamespace(entrypoint=mark,parameters={'properties':{}})
    tk.functions['replio_thread_ensure_support_task']=SimpleNamespace(entrypoint=own,parameters={'properties':{}})
   if case.get('block'):
    d._respond_results=[{'sent':False,'blocked':True,'retry_now':False,'category':'no_progress','reason':'Held for manual review.'}]*3
   model=StdioModel(command);started=time.monotonic();errors=[];output={}
   try:
    with dry_run_scope(True):
     result=await asyncio.wait_for(c.run(agent=SimpleNamespace(_mcp=pool,model=model),event={'slug':'replio-thread'},payload={'payload':{'thread_id':'sim-'+case['id'],'product':case['product'],'channel_kind':'email_imap','message':{'body_text':case['turns'][-1][1]}}},session_id='regression:'+case['id'],delivery_id='sim'),240)
    output=json.loads(result.text)
    import re
    reply=output.get('reply','')
    if not case.get('block') and (not reply or 'replio_threads_respond' not in d.names):errors.append('no delivered simulated reply')
    for term in case.get('required',[]):
     if not re.search(term,reply,re.I):errors.append('missing '+term)
    for term in case.get('forbidden',[]):
     if term.casefold() in reply.casefold():errors.append('forbidden '+term)
    if case.get('reclaimed') and not output.get('facts',{}).get('human_owner_verified'):errors.append('no verified support owner')
    if case.get('block') and output.get('facts',{}).get('delivery_state')!='blocked':errors.append('blocked send misreported')
   except Exception as e:errors.append(type(e).__name__+': '+str(e)[:200])
   row={'case':case['id'],'iteration':iteration,'seconds':round(time.monotonic()-started,1),'failures':errors,'turns':case['turns'],'output':output,'calls':model.calls,'tools':d.names}
   rows.append(row);print(json.dumps({k:row[k] for k in ('case','iteration','seconds','failures')}),flush=True)
   Path(args.output).write_text(json.dumps({'cases':len(rows),'passed':sum(not x['failures'] for x in rows),'business_io':'simulated only','rows':rows},ensure_ascii=False,indent=2))
 return sum(bool(x['failures']) for x in rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model-command-file',required=True);p.add_argument('--docs-command-file',required=True);p.add_argument('--voice-model',required=True);p.add_argument('--output',required=True);p.add_argument('--repeat',type=int,default=1);p.add_argument('--case',action='append');args=p.parse_args();raise SystemExit(bool(asyncio.run(run(args))))

#!/usr/bin/env python3
"""Two bounded training trials followed by matched screening and conditional validation."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/baseline_improve_20261008'
SOURCE=ROOT/'outputs/competition_grasp_weighted_continue_10k/checkpoints/global_step_6500'
PY_SIM='/home/zhongde/miniconda3/envs/RoboTwin/bin/python'

def announce(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'),message,flush=True)

def run_logged(cmd,path,env):
    announce(f'Start {path.name}: {cmd}')
    with path.open('a') as f:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    return result.returncode

def summarize(path,manifest):
    spec=importlib.util.spec_from_file_location('sweep',ROOT/'02_代码材料/diagnostics/checkpoint_sweep.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=m.aggregate(path,json.loads(manifest.read_text()))
    (path/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def train(name):
    out=BASE/name;out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    link=out/'checkpoints/global_step_6500'
    if not link.exists():link.symlink_to(SOURCE,target_is_directory=True)
    if not (out/'.trained').exists():
        lowmem=(out/'.lowmem').exists()
        for attempt in range(2):
            suffix='_lowmem' if lowmem else ''
            config=ROOT/f'02_代码材料/configs/train_baseline_{name}_6750{suffix}.yaml'
            log=out/f'train_{time.time_ns()}.log'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MASTER_PORT='62614',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            code=run_logged(['bash','train.sh','tasks/vla/train_lingbotvla.py',str(config)],log,env)
            if code==0:break
            content=log.read_text(errors='replace')
            oom=any(s in content for s in ['CUDA out of memory','torch.OutOfMemoryError','CUDA error: out of memory'])
            if not oom or lowmem:raise RuntimeError(f'Training failed: {log}; no blind retry')
            lowmem=True;(out/'.lowmem').touch()
            announce(f'{name}: CUDA OOM; retry once with horizon25 and one worker')
        ckpt=out/'checkpoints/global_step_6750/hf_ckpt'
        index=json.loads((ckpt/'model.safetensors.index.json').read_text())
        for shard in set(index['weight_map'].values()):
            if not (ckpt/shard).is_file():raise RuntimeError(f'Missing checkpoint shard {shard}')
        (out/'.trained').touch()
    announce(f'Training complete: {name}')
    return out/'checkpoints/global_step_6750/hf_ckpt'

def evaluate(label,model,manifest):
    out=BASE/label;out.mkdir(exist_ok=True)
    tasks=json.loads(manifest.read_text());counts={len(v) for v in tasks.values()}
    if len(counts)!=1:raise ValueError('Unequal episode counts')
    n=next(iter(counts))
    if not (out/'.complete').exists():
        env=dict(os.environ,MODEL_PATH=str(model),OUT_DIR=str(out),MANIFEST=str(manifest),TASKS=' '.join(tasks),EPISODES=str(n),TEMPORAL_ENSEMBLE='False',VIDEO='True',GUI='0',PORT='9360',ADAPTIVE_HORIZON='False',PER_EPISODE_SAMPLING_SEED='True',LINGBOT_FLOW_SOLVER='euler')
        code=run_logged(['bash','02_代码材料/diagnostics/run.sh','10'],out/'runner.log',env)
        if code:raise RuntimeError(f'Evaluation failed: {out}')
    result=summarize(out,manifest);(out/'.complete').touch()
    announce(f'{label}: {result["successes"]}/{result["total"]}')
    return result

def main():
    os.chdir(ROOT);BASE.mkdir(exist_ok=True)
    lock=(BASE/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (SOURCE/'optimizer').is_dir():raise RuntimeError('Baseline optimizer missing')
    if shutil.disk_usage(BASE).free<150*2**30:raise RuntimeError('Need at least150GiB free for two checkpoints and videos')
    screen=ROOT/'docs/results/phase_loss_20261008/manifest.json'
    baseline=json.loads((ROOT/'docs/results/phase_loss_20261008/baseline6500_euler.json').read_text())
    outcomes={};models={}
    for name in ['accum4','padonly']:
        models[name]=train(name)
        outcomes[name]=evaluate(f'screen_{name}',models[name],screen)
        (BASE/'screen_summary.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2))
    winner=max(outcomes,key=lambda k:outcomes[k]['successes'])
    decision={'baseline_screen':baseline['successes'],'best_trial':winner,'trial_screen':outcomes[winner]['successes'],'promoted':False}
    if outcomes[winner]['successes']>baseline['successes']:
        manifest=ROOT/'docs/results/checkpoint_validation_manifest.json'
        control=evaluate('validation_baseline6500',SOURCE/'hf_ckpt',manifest)
        candidate=evaluate(f'validation_{winner}',models[winner],manifest)
        decision.update(validation_baseline=control['successes'],validation_trial=candidate['successes'],validation_total=candidate['total'],note='Development validation only; no automatic baseline replacement')
    else:
        decision['note']='No screening improvement over baseline; stop without further training or large evaluation'
    (BASE/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    (BASE/'.complete').touch();announce(str(decision))

if __name__=='__main__':main()

import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

EXP=Path(__file__).resolve().parents[1]
PYTHON="/data/home/jiboya/.conda/envs/spamosaic-env/bin/python"
GPUS=[1,6]
ENV=dict(os.environ,R_HOME="/data/home/jiboya/.conda/envs/spamosaic-env/lib/R",
         LD_LIBRARY_PATH="/data/home/jiboya/.conda/envs/spamosaic-env/lib",
         OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
         NUMBA_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8",
         PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1")
LOCK=threading.Lock()

def execute(arguments,log,gpu):
    uuid=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    command=shlex.join([PYTHON]+arguments)+" 2>&1 | tee "+shlex.quote(str(log))
    subprocess.run(["bash","-o","pipefail","-c",command],env=dict(ENV,CUDA_VISIBLE_DEVICES=uuid),cwd=str(EXP.parents[1]),check=True,
                   stdout=subprocess.DEVNULL)

def phase(name,jobs,prepare=False):
    pending=queue.Queue()
    for job in jobs:
        marker = (EXP/"results/cache"/f'{job["dataset"]}_rep{job["rep"]}'/"provenance.json"
                  if prepare else EXP/"results/runs"/job["id"]/"complete.json")
        if marker.exists():
            print(name,"REUSE",job["id"],flush=True)
        else:
            pending.put(job)
    def worker(gpu):
        while True:
            with LOCK:
                if pending.empty(): return
            free=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),
                     "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
            if free < 20000:
                time.sleep(30)
                continue
            with LOCK:
                if pending.empty(): return
                job=pending.get_nowait()
            identity=job["id"]
            print(name,"START",identity,"GPU",gpu,flush=True)
            if prepare:
                args=[str(EXP/"scripts/prepare_data.py"),"--dataset",job["dataset"],"--rep",job["rep"],"--device","cuda:0"]
            else:
                args=[str(EXP/"scripts/run_exp004.py"),"--job",str(EXP/"scripts/jobs"/(identity+".json")),"--device","cuda:0"]
            execute(args,EXP/"log"/(identity+".log"),gpu)
            print(name,"DONE",identity,flush=True)
    (EXP/"results/status.json").write_text(json.dumps({"phase":name,"jobs":len(jobs),"started":time.time(),"gpus":GPUS,"max_concurrent_jobs":2}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for result in pool.map(worker,GPUS): pass

def validation_score(cfg):
    with (EXP/"results/runs"/cfg["id"]/"diagnostics.csv").open() as f:
        row=list(csv.DictReader(f))[-1]
    return (float(row["cosine_ab_mrr"])+float(row["cosine_ba_mrr"]))/2

def main():
    hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((EXP/"scripts").rglob("*")) if p.is_file() and "__pycache__" not in str(p)}
    (EXP/"results/source_sha256.json").write_text(json.dumps(hashes,indent=2))
    preparations=[dict(id=f"prepare_{ds}_rep{rep}",dataset=ds,rep=rep)
                  for ds,reps in (("Simulations",("1","2","3")),("MB-H3K4me3",("1",))) for rep in reps]
    phase("prepare",preparations,True)
    phase("pilot",json.loads((EXP/"results/pilot_manifest.json").read_text()))
    screen=json.loads((EXP/"results/screen_manifest.json").read_text())
    phase("screen",screen)
    selected=[]
    selection=[]
    for dataset in ("Simulations","MB-H3K4me3"):
        configs=[x for x in screen if x["dataset"]==dataset]
        chosen=[x for x in configs if x["name"] in ("factor_dot_cl_all_none","factor_dot_bpr_all_both","factor_dot_bpr_cross_both","reconstruction_only")]
        for family in ("paper","spectrum","reference"):
            candidates=[x for x in configs if x["family"]==family and x["score"]=="topola"]
            best=max(candidates,key=validation_score)
            chosen.append(best)
            if family=="paper":
                chosen.extend(x for x in candidates if x["lambda"]==best["lambda"] and x["negatives"]!=best["negatives"])
            if family=="reference":
                # Test the selected reference mechanism with the entire loss training set.
                complete=dict(best,fixed_query=False,query_size=2048 if dataset=="Simulations" else 1024,
                              reference="fixed",reference_size=0)
                complete["name"]=best["name"]+"_full_training"
                chosen.append(complete)
        for cfg in chosen:
            selection.append({"dataset":dataset,"name":cfg["name"],"source_id":cfg["id"],
                              "validation_cosine_mrr":validation_score(cfg),"configuration":cfg})
            for rep in (("1","2","3") if dataset=="Simulations" else ("1",)):
                for seed in (2024,2025,2026,2027):
                    job=dict(cfg,rep=rep,seed=seed,stage="confirm",
                             id=f'confirm_{dataset}_{cfg["name"]}_rep{rep}_seed{seed}')
                    (EXP/"scripts/jobs"/(job["id"]+".json")).write_text(json.dumps(job,indent=2))
                    selected.append(job)
    (EXP/"results/selection.json").write_text(json.dumps({
        "rule":"Maximum final held-out bidirectional cosine MRR within each family; no biological labels or iLISI used for selection.",
        "selected":selection},indent=2))
    (EXP/"results/confirm_manifest.json").write_text(json.dumps(selected,indent=2))
    phase("confirm",selected)
    rows=[]
    for job in screen+selected:
        run=EXP/"results/runs"/job["id"]
        for path in sorted(run.glob("metrics_e*.csv")):
            epoch,representation=path.stem.replace("metrics_e","").split("_",1)
            with path.open() as f:
                for row in csv.DictReader(f):
                    rows.append(dict(id=job["id"],stage=job["stage"],dataset=job["dataset"],rep=job["rep"],
                                     seed=job["seed"],name=job["name"],family=job["family"],
                                     epoch=epoch,representation=representation,**row))
    with (EXP/"results/all_metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (EXP.parents[1]/"experiments/index.csv").open("a") as f:
        f.write("exp004,2026-10-28,候选集合匹配CL/BPR;严格lambda扫描;尺度梯度校准;参考集合消融;双几何评估;300epochs;多种子与3模拟重复,完成;详见exp004_topola_mechanism/results/all_metrics.csv\n")
    (EXP/"results/status.json").write_text(json.dumps({"phase":"complete","screen":len(screen),"confirm":len(selected),"finished":time.time()}))
    print("EXPERIMENT COMPLETE",flush=True)

if __name__=="__main__":main()

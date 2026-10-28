import itertools
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

def configurations(dataset):
    t=.1 if dataset=="Simulations" else .01
    base=dict(dataset=dataset,rep="1",seed=1234,epochs=300,score="dot",
              objective="cl",negatives="all",temperature=t,temperature_multiplier=1.,
              alpha=1.,calibration="none",reference="query",reference_size=0,
              lambda_=10.,size_scaled=False,query_size=2048 if dataset=="Simulations" else 1024,
              fixed_query=False,holdout=True)
    base["lambda"]=base.pop("lambda_")
    rows=[]
    def add(name, family, **changes):
        cfg=dict(base,**changes)
        cfg.update(name=name,family=family,id=f'{dataset}_{name}_rep1_seed1234',stage="screen")
        rows.append(cfg)
    for score,objective,negative in itertools.product(("dot","topola"),("cl","bpr"),("all","cross")):
        for calibration in ("none","both"):
            add(f"factor_{score}_{objective}_{negative}_{calibration}","factor",
                score=score,objective=objective,negatives=negative,calibration=calibration)
    for negative,lam in itertools.product(("all","cross"),(1e-7,1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,.1,1.)):
        add(f"paper_{negative}_{lam:g}","paper",score="topola",objective="bpr",negatives=negative,
            **{"lambda":lam},temperature=lam)
    for objective,lam in itertools.product(("cl","bpr"),(.001,1.,30.,100.,300.)):
        add(f"spectrum_{objective}_{lam:g}","spectrum",score="topola",objective=objective,
            calibration="both",**{"lambda":lam})
    for reference,size in (("query",256),("fixed",512),("fixed",1024),("fixed",0),
                           ("random",512),("random",1024),("spatial",512),("spatial",1024)):
        for scaled in (False,True):
            add(f"reference_{reference}_{size}_{int(scaled)}","reference",score="topola",
                objective="bpr",reference=reference,reference_size=size,query_size=256,
                fixed_query=True,size_scaled=scaled,calibration="both" if scaled else "none",
                **{"lambda":10.},temperature=.001)
    add("query256_dot_control","reference",objective="bpr",query_size=256,fixed_query=True,calibration="both")
    add("reconstruction_only","control",alpha=0.)
    add("dot_bpr_T1","control",objective="bpr",temperature=1.)
    add("allbridge_dot_cl","control",holdout=False)
    add("allbridge_dot_bpr","control",holdout=False,objective="bpr")
    return rows

def main():
    jobs=EXP/"scripts/jobs"
    jobs.mkdir(exist_ok=False)
    manifest=[]
    for dataset in ("Simulations","MB-H3K4me3"):
        for cfg in configurations(dataset):
            path=jobs/(cfg["id"]+".json")
            path.write_text(json.dumps(cfg,indent=2))
            manifest.append(cfg)
    (EXP/"results/screen_manifest.json").write_text(json.dumps(manifest,indent=2))
    pilots=[]
    for dataset in ("Simulations","MB-H3K4me3"):
        for name in ("factor_topola_bpr_all_both","reference_fixed_0_1"):
            cfg=next(x.copy() for x in manifest if x["dataset"]==dataset and x["name"]==name)
            cfg.update(id="pilot_"+cfg["id"],epochs=2,stage="pilot")
            (jobs/(cfg["id"]+".json")).write_text(json.dumps(cfg,indent=2))
            pilots.append(cfg)
    (EXP/"results/pilot_manifest.json").write_text(json.dumps(pilots,indent=2))
    print("screen",len(manifest),"pilot",len(pilots))

if __name__=="__main__":main()

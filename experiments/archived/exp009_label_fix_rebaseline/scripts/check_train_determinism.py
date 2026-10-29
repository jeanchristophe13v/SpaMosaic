"""确定性排查 2:同一进程内,相同预处理结果上把同一 run(Simulations dot / topola_cal, seed 1234)训练两次,比较指标。"""
import argparse
import sys
from os.path import abspath, dirname

sys.path.insert(0, dirname(abspath(__file__)))
import run_exp009 as drv

args = argparse.Namespace(sim_setting='rna-0.1-12-adt-2.1', sim_rep='1', device=sys.argv[1], epochs=100,
                          lambda_topola=10.0, out_dir='/tmp/exp009_det_runs', seed=1234)
input_dict, raw, layout = drv.load_inputs('Simulations', args)
for variant in ('dot', 'topola_cal'):
    res = [drv.run_arm('Simulations', variant, input_dict, raw, layout, args)[0] for _ in range(2)]
    d = max(abs(res[0][k] - res[1][k]) for k in res[0])
    print(f'[det] {variant}: 同进程两次训练的最大指标差 = {d:.3e}; iLISI {res[0][("iLISI","all")]:.4f} / {res[1][("iLISI","all")]:.4f}', flush=True)

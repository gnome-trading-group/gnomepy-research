import sys, time, pandas as pd
sys.path.insert(0, 'gnomepy_research/models/cs2/inplay/free/work')
from exp import load, walk, cmp, C
t=time.time()
rt, st = load()
r = walk(rt, st, ['2026-08'])
r.to_parquet(f'{C}/rep_aug.parquet')
bm = pd.read_parquet('/Users/mprey/gnome-research-work/cs2/comp3/bench_cache/bench_89bd1ebdec_2026-08.parquet')
bm['month']='2026-08'
print(len(r), len(bm))
cmp(r, bm, 'replica vs bench')
print(time.time()-t)

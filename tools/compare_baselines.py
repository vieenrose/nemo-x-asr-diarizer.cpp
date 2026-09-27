#!/usr/bin/env python3
# Score three systems' transcripts on the 12 eval clips: WER + right/wrong speaker share (score_stream.py).
# Usage: tools/compare_baselines.py <dir>  - expects <dir>/{v15_,v7c_,dev_25_}<clip>.txt (7B tags stripped).
S=sys.argv[1]; systems=[("VibeASR 1.5B (phone)","v15_"),("VibeASR 7B (host)","v7c_"),("composite (phone)","dev_25_")]
CL="gate_ms gate_ms_g100 gate_ms_g1000 gate_ms_v2 control_ls holdout_en holdout_en_aligned holdout_zh holdout_zh2 holdout_zh_aligned holdout_zh_ph35200 gate_long".split()
M={'gate_long':'manifest_long','gate_ms':'manifest_ms','gate_ms_g100':'manifest_ms_g100','gate_ms_g1000':'manifest_ms_g1000','gate_ms_v2':'manifest_ms_v2'}
MULTI={'gate_ms','gate_ms_g100','gate_ms_g1000','gate_ms_v2','gate_long'}
tot={p:[0,0] for _,p in systems}; att={p:[0,0,0] for _,p in systems}
print(f"{'clip':20s}"+"".join(f"{n:>24s}" for n,_ in systems))
for c in CL:
    man=json.load(open(f"../eval-bilingual/{M.get(c,'manifest_'+c)}.json")); row=f"{c:20s}"
    for n,p in systems:
        f=f"{S}/{p}{c}.txt"
        
        try: r=ss.score(f,man)
        except Exception as e: row+=f"{'n/a':>24s}"; continue
        e=r['wer']; tot[p][0]+=e*len(ss.ref_stream(man)); tot[p][1]+=len(ss.ref_stream(man))
        cell=f"WER {e:.3f}"
        if c in MULTI:
            a,cov=r['attribution'],r['coverage']; right=cov*(1-a); wrong=cov*a
            att[p][0]+=right; att[p][1]+=wrong; att[p][2]+=1
            cell+=f" ✓{right*100:.0f}/✗{wrong*100:.0f}%"
        row+=f"{cell:>24s}"
    print(row)
print(f"{'MICRO WER':20s}"+"".join(f"{t[0]/t[1]:>24.4f}" if t[1] else f"{'':>24s}" for t in tot.values()))
print(f"{'mean right/wrong spk':20s}"+"".join(f"{'%.0f%% / %.0f%%'%(a[0]/a[2]*100,a[1]/a[2]*100) if a[2] else '':>24s}" for a in att.values()))

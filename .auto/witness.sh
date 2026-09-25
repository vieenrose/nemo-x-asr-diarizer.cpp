#!/bin/bash
# usage: witness.sh <snap_before> <snap_after>  -- snaps are `cat /sys/.../cpu6/cpufreq/stats/time_in_state`
python3 - "$1" "$2" <<'PY'
import re,sys
def parse(t):
    o={}
    for ln in t.splitlines():
        m=re.match(r'\s*(\d+)\s+(\d+)',ln)
        if m: o[int(m.group(1))]=int(m.group(2))
    return o
a,b=parse(open(sys.argv[1]).read()),parse(open(sys.argv[2]).read()); tot=dt=0.0
for f in sorted(set(a)&set(b)):
    d=b[f]-a[f]
    if d>0: tot+=d*f; dt+=d
mhz=tot/dt/1000 if dt else 0
pct=100*sum(b[f]-a[f] for f in set(a)&set(b) if f>=2400000 and b[f]>a[f])/dt if dt else 0
print("witness mean %.1f MHz, %%.time>=2.4GHz %.1f -> %s" % (mhz,pct,"ARMED" if mhz>=1950 else "UNARMED: refuse to trust this data"))
PY

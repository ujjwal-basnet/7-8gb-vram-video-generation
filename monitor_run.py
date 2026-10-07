"""Launch only the render worker; measure GPU/device RAM and protect host RAM."""
import subprocess
import sys
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent
log=root/"render.log"
with log.open("w") as stream:
    worker=subprocess.Popen([sys.executable,str(root/"main.py"),*sys.argv[1:]],stdout=stream,stderr=subprocess.STDOUT)
    measurements=dict(worker_pid=worker.pid,peak_worker_ram_gib=0,peak_device_used_gib=0,
                      minimum_available_ram_gib=None,memory_guard_stopped=False)
    while worker.poll() is None:
        try:
            status=Path(f"/proc/{worker.pid}/status").read_text().splitlines()
            rss=int(next(x for x in status if x.startswith("VmRSS:")).split()[1])/2**20
            mem=Path("/proc/meminfo").read_text().splitlines()
            available=int(next(x for x in mem if x.startswith("MemAvailable:")).split()[1])/2**20
            measurements["peak_worker_ram_gib"]=max(measurements["peak_worker_ram_gib"],rss)
            measurements["minimum_available_ram_gib"]=min(measurements["minimum_available_ram_gib"] or available,available)
            used=float(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).splitlines()[0])/1024
            measurements["peak_device_used_gib"]=max(measurements["peak_device_used_gib"],used)
            if available < 1 and rss > 6:
                measurements["memory_guard_stopped"]=True
                worker.terminate()
        except (FileNotFoundError,StopIteration,ProcessLookupError):
            pass
        (root/"memory-observation.json").write_text(json.dumps(measurements,indent=2))
        time.sleep(1)
    measurements["worker_exit_code"]=worker.wait()
    (root/"memory-observation.json").write_text(json.dumps(measurements,indent=2))
sys.exit(measurements["worker_exit_code"])

"""Synchronized stage timings; nested stages overlap and must not be summed."""
from contextlib import contextmanager
import json
import time


class StageTimings:
    def __init__(self, path):
        self.path = path
        self.events = []

    @contextmanager
    def measure(self, name):
        import torch
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.events.append(dict(stage=name, seconds=time.perf_counter() - started,
                                    cumulative_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                    allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                    reserved_gib=torch.cuda.memory_reserved() / 2**30))
            self.path.write_text(json.dumps(self.events, indent=2))
            print(f"TIMING {name}: {self.events[-1]['seconds']:.2f}s", flush=True)

    def wrap(self, function, name):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured

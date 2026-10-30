"""Read-only NVML timeline, explicitly separate from controller observations."""
import json
import signal
import sys
import time

import pynvml
from mpc_lab.runtime import local_clock_domain


def main(path):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        with open(path,"x") as output:
            while not stopped:
                begin = time.monotonic()
                row = dict(clock_domain=local_clock_domain(),sample_begin=begin,
                    sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_SM),
                    memory_clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_MEM),
                    power_w=pynvml.nvmlDeviceGetPowerUsage(handle)/1000,
                    temperature_c=pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU),
                    gpu_utilization=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    memory_used_bytes=pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                row["sample_end"] = time.monotonic()
                output.write(json.dumps(row,allow_nan=False)+"\n")
                output.flush()
                time.sleep(max(0,1-(time.monotonic()-begin)))
    finally:
        pynvml.nvmlShutdown()


if __name__=="__main__":
    main(sys.argv[1])

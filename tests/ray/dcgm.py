# dictionary_reader_example.py
import sys
import pynvml

sys.path.append("/usr/local/dcgm/bindings/python3")

from DcgmReader import DcgmReader
import dcgm_fields
import time

# dr = DcgmReader(fieldIds=[dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION], updateFrequency=100)

# start_time = time.perf_counter()
# data = dr.GetLatestGpuValuesAsDict(True)
# end_time = time.perf_counter()
# print(data, end_time-start_time)
# for gpuId in data:
#     print(data[gpuId], type(data[gpuId]))
#     for fieldTag, val in data[gpuId].items():
#         msg = "GPU:%s:%s=%s" % (str(gpuId), fieldTag, val)
#         print(msg)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
start_time = time.perf_counter()
power = pynvml.nvmlDeviceGetPowerUsage(handle)
end_time = time.perf_counter()
print(power, end_time-start_time)
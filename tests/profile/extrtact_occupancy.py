from multiprocessing import pool
import numpy as np
import os
import json
from tqdm import tqdm
from hfutils.plot import sns_displot, distplot
import seaborn as sns
import matplotlib.pyplot as plt
from hfutils.cuda import CudaOccupancyCalculator


calculator = CudaOccupancyCalculator("8.0")

batches = [1,2,4,8,16,32,64,128]
model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]

def plot_density(data, filename):
    sns.distplot(data, hist=True, kde=False, 
                bins=int(180/5), 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2})
    plt.savefig(os.path.join("figures", "single_profile", filename), bbox_inches="tight")
    plt.close()

# data = {}

# np.random.seed(1234567890)

for bsz in batches:
    for name in model_names:
        key = f"{name}_{bsz}"

        trace_name_base = f"trace_{name}_{bsz}"
        dur_name_base = f"dur_{name}_{bsz}"
        trace_raw = f"{trace_name_base}.json"
        trace_fig = f"{trace_name_base}.png"
        dur_fig = f"{dur_name_base}.png"
        trace_occ = f"{trace_name_base}.npy"
        dur_occ = f"{dur_name_base}.npy"
        with open(os.path.join("data", "single_profile", trace_raw), "r") as fp:
            trace = json.load(fp)
            trace_events = trace["traceEvents"]

            occupancy = []
            duration = []

            for event in tqdm(trace_events, desc=key):
                # if "dur" in event and "ts" in event:
                #     duration.append([event['dur'],event['ts']])
                args = event.get('args', None)
                if args is None: continue
                if "est. achieved occupancy %" in args:
                    calculator.set_inputs(
                        args['block'][0],
                        args['registers per thread'],
                        "8.0",
                        args['shared memory']
                    )

                    occupancy.append(calculator.occupancyOfMultiprocessor())
                    duration.append([event['dur'],event['ts']])
            
            # data[key] = occupancy
            # plot_density(occupancy, trace_fig)
            # plot_density(duration, dur_fig)
            # occupancy = np.array(occupancy)
            duration = np.array(duration)
            # occupancy = occupancy[occupancy[:, -1].argsort()]
            duration = duration[duration[:, -1].argsort()]
            print(
                key,
                duration[-1, -1] - duration[0, -1],
                np.sum(duration[:, 0]) / (duration[-1, -1] - duration[0, -1]),
                np.sum(duration[:, 0])
            )
            
            np.save(os.path.join("data", "single_profile", trace_occ), occupancy, allow_pickle=False)
            np.save(os.path.join("data", "single_profile", dur_occ), duration, allow_pickle=False)
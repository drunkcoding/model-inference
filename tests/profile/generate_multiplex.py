import functools
from multiprocessing import pool
import numpy as np
import os
import json
from tqdm import tqdm
from hfutils.plot import sns_displot, distplot
from sklearn.metrics import r2_score

batches = [1,2,4,8,16,32,64,128]
model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]
model_keys = [
    "S",
    "M",
    "L",
    "XL"
]

data_trace = {}
data_kdur = {}
data_delay = {}

np.random.seed(1234567890)

for bsz in batches:
    for i, name in enumerate(model_names):
        key = f"{model_keys[i]}_{bsz}"
        filename = os.path.join("data", "single_profile", f"trace_{name}_{bsz}.npy")
        data_trace[key] = np.load(filename, allow_pickle=False)
        filename = os.path.join("data", "single_profile", f"dur_{name}_{bsz}.npy")
        data_kdur[key] = np.load(filename, allow_pickle=False)

        with open(os.path.join("data", "single_model", f"ray-{model_keys[i]}-{bsz}.json"), "r") as fp:
            base = json.load(fp)
            base_t = []
            for v in base.values():
                base_t += v 

        data_delay[key] = base_t

def run_search(a,b, dur_a, dur_b):
    # if len(a) < len(b):
    #     a, b = b, a
    #     dur_a, dur_b = dur_b, dur_a

    len_a = len(a)
    len_b = len(b)


    # data = []
    # for _ in range(40):
    #     start_pos = np.random.randint(0, len_a - len_b) if len_a != len_b else 0
        
    #     temp_a = a[start_pos:(start_pos + len_b)]
    #     temp_b = b

    #     temp_dur_a = dur_a[start_pos:(start_pos + len_b)]
    #     temp_dur_b = dur_b

    #     # print("====", len(temp_a), len(temp_b), len(temp_dur_a), len(temp_dur_b))

    #     sum_arr = (temp_a + temp_b) > 1
    #     data.append(
    #         np.sum(((temp_dur_a + temp_dur_b) / 2)[sum_arr]) / 1000
    #     )
    #     # print(len(sum_arr), np.count_nonzero(~sum_arr))
    #     # data += ((temp_dur_a + temp_dur_b) / 2)[sum_arr].tolist() + [0] * np.count_nonzero(~sum_arr)
    data = []
    length = max(len_a, len_b)
    a_index = [x for x in range(len_a)]
    b_index = [x for x in range(len_b)]
    for _ in range(20):
        sample_a_index = np.random.choice(a_index, length)
        sample_b_index = np.random.choice(b_index, length)
        sample_a = a[sample_a_index]
        sample_b = b[sample_b_index]
        sample_a_dur = dur_a[sample_a_index]
        sample_b_dur = dur_b[sample_b_index]

        sum_arr = (sample_a + sample_b) > 1

        data.append(
            np.sum(((sample_a_dur + sample_b_dur) / 2)[sum_arr]) / 1000
        )
    
    return data

data_conv = {}
for key_i in tqdm(data_trace):
    for  key_j in tqdm(data_trace):
        # sample_i = np.array(data_trace[key_i])
        # sample_j = np.array(data_trace[key_j])

        combined_key = key_i + "_" + key_j

        data_conv[combined_key] = run_search(
            data_trace[key_i],
            data_trace[key_j],
            data_kdur[key_i],
            data_kdur[key_j],
        )

        np.save(os.path.join("data", "combined_model", f"{combined_key}.npy"), data_conv[combined_key])
        distplot(os.path.join("figures", "combined_model", f"{combined_key}.png"), data_conv[combined_key])




from fileinput import filename
import functools
from multiprocessing import pool
import numpy as np
import os
from tqdm import tqdm
from hfutils.plot import sns_displot, distplot
import multiprocessing as mp

batches = [1,2,4,8,16,32,64,128]
model_names = [
    "t5-small-lm-adapt",
    "t5-base-lm-adapt",
    "t5-large-lm-adapt",
    "t5-xl-lm-adapt",
]

data = {}

np.random.seed(1234567890)

for bsz in batches:
    for name in model_names:
        key = f"{name}_{bsz}"
        filename = os.path.join("data", f"trace_{name}_{bsz}.npy")
        data[key] = np.load(filename, allow_pickle=False)

def run_search(skip, a, b):
    data = []

    len_a = len(a)
    len_b = len(b)

    k_a = 0
    k_b = 0
    while k_a + skip < len_a and k_b < len_b:
        data.append(a[k_a+skip] + b[k_b])
        k_a += 1
        k_b += 1

    while k_b < len_b:
        data.append(b[k_b])
        k_b += 1

    while k_a + skip < len_a:
        data.append(a[k_a+skip])
        k_a += 1

    return data

def sliding_window_sum(a,b):
    len_a = len(a)
    len_b = len(b)

    data = []

    # pool = mp.Pool(20)
    # results = pool.map(functools.partial(run_search, a=a, b=b), [skip for skip in range(0,len_a,2)])

    # for r in results:
    #     data += r

    length = max(len_a, len_b)
    for _ in range(20):
        sample_a = np.random.choice(a, length)
        sample_b = np.random.choice(b, length)

        data += (sample_a + sample_b).tolist()

    
    # for skip in tqdm(range(len_a)):
    #     k_a = 0
    #     k_b = 0
    #     while k_a + skip < len_a and k_b < len_b:
    #         data.append(a[k_a+skip] + b[k_b])
    #         k_a += 1
    #         k_b += 1

    #     while k_b < len_b:
    #         data.append(b[k_b])
    #         k_b += 1

    #     while k_a + skip < len_a:
    #         data.append(a[k_a+skip])
    #         k_a += 1

    return data

data_conv = {}
for key_i in tqdm(data):
    for  key_j in tqdm(data):
        # sample_i = np.random.choice(data[key_i], len(data[key_i]) // 10)
        # sample_j = np.random.choice(data[key_j], len(data[key_j]) // 10)

        sample_i = np.array(data[key_i])
        sample_j = np.array(data[key_j])

        sample_i = sample_i[sample_i != 0]
        sample_j = sample_j[sample_j != 0]

        # sample_i = data[key_i]
        # sample_j = data[key_j]
        
        combined_key = key_i + ":" + key_j
        data_conv[combined_key] = sliding_window_sum(sample_i, sample_j)

        print(key_i, key_j, len(sample_i), len(sample_j))
        
        np.save(os.path.join("data", f"{key_i}_{key_j}.npy"), data_conv[combined_key])
        distplot(os.path.join("figures", f"{key_i}_{key_j}.png"), data_conv[combined_key])

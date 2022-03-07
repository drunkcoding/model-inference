from argparse import Namespace
from os import times
import requests
import multiprocessing as mp
from tqdm import tqdm
import time
from scipy import stats
import ray
from ray import serve

NUM_PROC = 4
barrier = mp.Barrier(NUM_PROC)
URL = "http://127.0.0.1:8000/echo"

# ray.init(Namespace="echo")

def test_body(pid):
    print(pid)

    # handle = serve.get_deployment("echoserver").get_handle(sync=True)

    time_list = []
    for _ in tqdm(range(1000)):
        start_time = time.perf_counter()
        resp = requests.post(URL, json={"payload": [1.234] * 300})
        # handle.remote()
        resp = resp.json()
        end_time = time.perf_counter()
        time_list.append((end_time-start_time)*1000)
    print(stats.describe(time_list))   
    return pid

pool = mp.Pool(processes=NUM_PROC)
pool.map(
    test_body,
    [i for i in range(NUM_PROC)],
)
pool.close()
pool.join()


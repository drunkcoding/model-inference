import json
import os, re
import numpy as np
import pandas as pd
from scipy.stats import describe
from tqdm import tqdm

# Open a file
path = "/home/ubuntu/model-inference/rayserve/gpt-optim"
dirs = os.listdir(path)

# This would print all the files and directories
latency_list = []

requests = {}

for file in dirs:
    # print(file)
    filepath = os.path.join(path, file)
    if not "e" in file:
        continue
    print(filepath)
    with open(filepath, "r") as fp:
        text = fp.read()

    groups = re.findall(
        r"\[(.*),.*\].*\((\d+)\) inference \((.*), (.*), (.*), (.*)\)", text
    )
    # print(groups)
    for group in groups:
        uuid, bsz, start_time, end_time, start_power, end_power = group
        bsz = int(bsz)
        start_time = float(start_time)
        end_time = float(end_time)
        start_power = int(start_power)
        end_power = int(end_power)

        if not uuid in requests:
            requests[uuid] = []

        requests[uuid].append(
            {
                "uuid": uuid,
                "bsz": bsz,
                "start_time": start_time,
                "end_time": end_time,
                "start_power": start_power,
                "end_power": end_power,
            }
        )

agg_requests = {}
for uuid in requests:
    starty_time = min(record['start_time'] for record in requests[uuid])
    end_time = max(record['start_time'] for record in requests[uuid])
    bsz = requests[uuid][0]['bsz']
    agg_requests[uuid] = {
                "uuid": uuid,
                "bsz": bsz,
                "start_time": start_time,
                "end_time": end_time,
            }

latency_list = [
    sum(
        record["end_time"] - record["start_time"]
        for record in records
        if record["bsz"] == 1
    )
    for uuid, records in requests.items()
]

print(np.percentile(latency_list, 99))
print(np.percentile(latency_list, 50))
print(describe(latency_list))

time_list = np.array(
    [
        [record["end_time"], record["start_time"]]
        for uuid, records in requests.items()
        for record in records
        if record["bsz"] == 2
    ]
)

# df_time = pd.DataFrame(
#     [
#         record
#         for uuid, records in requests.items()
#         for record in records
#         if record["bsz"] == 2
#     ]
# )

df_time = pd.DataFrame(
    list(agg_requests.values())
)

print(df_time)

df_time = df_time.sort_values(by="start_time")
counts = [0 for _ in range(len(df_time.index))]

min_time = df_time["start_time"].min()
max_time = df_time["end_time"].max()

max_count = 0
max_records = None
TIME_WINDOW = 1.0
# max_interval = None
for t in tqdm(df_time["start_time"].to_numpy()):
    win_l = t
    win_h = t + TIME_WINDOW
    tmp_records = []
    tmp_counts = 0
    for idx, row in df_time.iterrows():
        # if (win_l <= row["end_time"] <= win_h) or (win_l <= row["start_time"] <= win_h):
        #     tmp_records.append(row)
        if row["end_time"] <= win_h and win_l <= row["start_time"]:
            tmp_counts += 1
            tmp_records.append(row)
            # print("enclosed", row, (win_l, win_h))
            print("encolsed", tmp_counts)
        elif row["end_time"] > win_h and row["start_time"] < win_h:
            tmp_counts += (win_h - row["start_time"]) / (
                row["end_time"] - row["start_time"]
            )
            tmp_records.append(row)
            print("high", tmp_counts)
            # print("high", row, (win_l, win_h))
        elif row["end_time"] > win_l and win_l > row["start_time"]:
            tmp_counts += (row["end_time"] - win_l) / (
                row["end_time"] - row["start_time"]
            )
            tmp_records.append(row)
            print("low", tmp_counts)
            # print("low", row, (win_l, win_h))

    if tmp_counts > max_count:
        max_count = tmp_counts
        max_records = tmp_records

    print("tmp_counts", tmp_counts)

print("max_count", max_count / TIME_WINDOW * 2)
# print("max_records", max_records)

# ts_list  = []
# power_list = []
# labels = []
# for row in max_records:
#     ts_list.append(row['start_time'])
#     ts_list.append(row['end_time'])

#     power_list.append(row['start_power'])
#     power_list.append(row['end_power'])

#     labels += [1, -1]

# df_energy = pd.DataFrame({
#     "ts": ts_list,
#     "power": power_list,
#     "labels": labels,
# })

# df_energy = df_energy.sort_values(by="ts")

# energy = (df_energy.ts - df_energy.ts.shift(1)) * (df_energy.power + df_energy.power.shift(1)) / 2
# energy = energy.to_numpy()
# labels = df_energy.labels.to_numpy()

# # print()
# count = 1
# e_sum = 0
# for i in range(1, len(labels)):
#     if count > 0: e_sum += energy[i]
#     count += labels[i]

# print("energy", e_sum / 1000 / len(max_records) / 2)

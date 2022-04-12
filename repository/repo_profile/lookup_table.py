import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

base_dir = os.path.dirname(__file__)

df_echo = pd.read_csv(os.path.join(base_dir, "profile_echo.csv"))
print(df_echo.describe())

data_size = df_echo['data_size'].to_numpy()

df_echo_latency = df_echo.groupby("data_size", as_index=False).agg({"latency": "mean"})
print(df_echo_latency)

df_numa = pd.read_csv(os.path.join(base_dir, "profile_same_numa.csv"))
print(df_numa.describe())

df_numa_latency = df_numa.groupby("data_size", as_index=False).agg({"latency": "mean"})
df_numa_latency['latency'] -= df_echo_latency['latency']
# print(df_numa_latency)

parameters = {}

x = df_numa_latency['data_size'].to_numpy()
y = df_numa_latency['latency'].to_numpy()
params = np.polyfit(x,y,1)
func = np.poly1d(params)
coefficient_of_dermination = r2_score(y, func(x))

parameters["numa"] = params.tolist()
print("df_numa_latency", coefficient_of_dermination, params, func(x))

df_cross = pd.read_csv(os.path.join(base_dir, "profile_cross_numa.csv"))
print(df_cross.describe())

df_cross_latency = df_cross.groupby("data_size", as_index=False).agg({"latency": "mean"})
df_cross_latency['latency'] -= df_echo_latency['latency']
# print(df_cross_latency)

x = df_cross_latency['data_size'].to_numpy()
y = df_cross_latency['latency'].to_numpy()
params = np.polyfit(x,y,1)
func = np.poly1d(params)
coefficient_of_dermination = r2_score(y, func(x))

parameters["cross"] = params.tolist()
print("df_cross_latency", coefficient_of_dermination, params, func(x))
# plt.savefig("df_cross_latency.png")

with open(os.path.join("tests/profile", "profile.json"), "w") as fp:
    json.dump(parameters, fp)
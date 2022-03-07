import copy
import os
import traceback
import numpy as np
import json
import pandas as pd
from collections import Counter

from tqdm import tqdm

batches = [1,2,4,8,16,32,64,128]


NUM_MODEL = 4
NUM_GPU = 2
NUM_STAGE = 2
NUM_REPLICATION = 4
LATENCY_BOUND = 2000

ratio = [1.0, 0.2, 0.05, 0.05]
inf_latency = [28, 46, 88, 102]
com_latency = [3, 4, 8, 12]
model_mem = [267, 928, 2.92 * 1024, 10.3 * 1024]
gpu_mem = [25 * 1024] * NUM_GPU
num_replica = [20, 10, 5, 2]


def dict_flatten(data):
    target = []
    for v in data.values():
        target += v

    return target

class CostModel:
    single_measure = None
    replica_measure = None
    gpu_occupancy = None

    relation = None

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

    batches = [1,2,4,8,16,32,64]

    num_model_type = len(model_names)

    def __init__(self, data_path_base) -> None:
        single_measure_path = os.path.join(data_path_base, "single_model")
        replica_measure_path = os.path.join(data_path_base, "replica_model")
        gpu_occupancy_path = os.path.join(data_path_base, "single_profile")

        self.single_measure = {i: dict() for i in range(len(self.model_keys))}
        self.replica_measure = {i: dict() for i in range(len(self.model_keys))}
        self.gpu_occupancy = {i: dict() for i in range(len(self.model_keys))}
        self.estimation = {i: dict() for i in range(len(self.model_keys))}

        for m in range(self.num_model_type):
            id = self.model_keys[m]
            name = self.model_names[m]
            for bsz in self.batches:
                with open(os.path.join(single_measure_path, f"ray-{id}-{bsz}.json"),'r') as fp:
                    # self.single_measure[(id,bsz)] = json.load(fp)
                    data = json.load(fp)
                    self.single_measure[m][bsz] = dict_flatten(data)

                    # print(self.single_measure.keys())
                    # print(self.single_measure[0].keys())
                    # print(self.single_measure[0][1].keys())

                with open(os.path.join(replica_measure_path, f"ray-{id}-R2-{bsz}.json"), "r") as fp:
                    # self.replica_measure[(id,bsz)] = json.load(fp)
                    data = json.load(fp)
                    self.replica_measure[m][bsz] = dict_flatten(data)

                    # print(self.replica_measure.keys())
                    # print(self.replica_measure[0].keys())
                    # print(self.replica_measure[0][1].keys())

                occupancy = np.load(os.path.join(gpu_occupancy_path,f"trace_{name}_{bsz}.npy"), allow_pickle=False)
                occupancy = occupancy[occupancy != 0]
                # self.gpu_occupancy[(id,bsz)] = np.mean(occupancy)
                self.gpu_occupancy[m][bsz] = occupancy

        self.model_occupancy = {
            m: [np.mean(value) for _, value in self.gpu_occupancy[m].items()] for m in range(self.num_model_type)
        }
        # print(self.single_measure[0][1].keys())
        self.model_density = {
            m: [
                np.log(len(value) / np.mean(self.single_measure[m][bsz]) * 1000)
                for bsz, value in self.gpu_occupancy[m].items()
            ] for m in range(self.num_model_type)
        }

    def fit(self):
        true_latency = []
        pred_latency = []

        for m in range(self.num_model_type):
            for b, bsz in enumerate(self.batches):
                true_latency.append(
                    np.abs(
                        np.mean(self.replica_measure[m][bsz])- np.mean(self.single_measure[m][bsz])
                    )
                )
                pred_latency.append(
                    self.model_occupancy[m][b] * 2
                    * bsz * self.model_density[m][b] # * np.mean(self.replica_measure[key])
                )
        # print(len(pred_latency), len(true_latency))
        self.relation = np.poly1d(np.polyfit(pred_latency,true_latency,1))
        self.occupancy = {
            m: np.poly1d(np.polyfit(self.batches,self.model_occupancy[m],1))
            for m in range(self.num_model_type)
        }
        self.density = {
            m: np.poly1d(np.polyfit(self.batches,self.model_density[m],1))
            for m in range(self.num_model_type)
        }
        self.latency = {
            m: np.poly1d(np.polyfit(self.batches, [np.mean(value) for value in self.single_measure[m].values()] , 1))
            for m in range(self.num_model_type)
        }

    def predict(self, models, ratio):
        occupancy = 0
        for m, r, s in models:
            bsz = ratio[m] * 64
            occupancy += self.occupancy[m](bsz) * bsz * self.density[m](bsz) / NUM_STAGE * s

        return self.relation(occupancy)


# MAX NUM_STAGE = NUM_GPU
# deployment = np.zeros((NUM_MODEL, NUM_GPU, NUM_REPLICATION))

def sequence_generate():
    seq = [ 0 for _ in range(NUM_STAGE) ]
    while True:
        yield copy.deepcopy(seq)
        cur = seq[0] + 1
        for i in range(NUM_STAGE):
            seq[i] = cur % NUM_GPU
            if (i+1) < NUM_STAGE:
                seq[i+1] += int(np.floor(cur / NUM_GPU))
                cur = seq[i+1]
        seq[-1]  = seq[-1] %  NUM_GPU
        # print("seq", seq)

def get_all_deployment():
    all_deployment = []
    all_gpu_group = []

    deployment = np.zeros((NUM_MODEL, NUM_REPLICATION, NUM_STAGE)).astype(int)
    
    all_gen = [[sequence_generate() for _ in range(NUM_MODEL)] for _ in range(NUM_REPLICATION)]
    
    def deployment_generator(m, r, counters):
        # gen = sequence_generate()
        # print("m", m, r)
        # deployment[m][r] = next(gen)
        while True:
            # if m == 0 and r == 0:
            #     counters = [Counter() for _ in range(NUM_GPU)]
            deployment[m][r] = next(all_gen[m][r])
            seq_sum = np.sum(deployment[m][r])

            local_counters = [Counter() for _ in range(NUM_GPU)]
            for g in deployment[m][r]:
                local_counters[g].update({(m,r): 1})

            # try:
            #     deployment[m][r] = next(all_gen[m][r])
            # except:
            #     traceback.print_stack()
            #     # break

            # print(deployment[m][r], m, r, seq_sum, (NUM_GPU-1) * NUM_STAGE)

            if m+1 < NUM_MODEL:
                deployment_generator(m+1, r, [counters[g] + local_counters[g] for g in range(NUM_GPU)])
            if r+1 < NUM_REPLICATION:
                deployment_generator(m, r+1, [counters[g] + local_counters[g] for g in range(NUM_GPU)])

                if r != 0:
                    deployment[m][r] =  [ -1 for _ in range(NUM_STAGE) ]

            if m == NUM_MODEL - 1:
                # print("deployment", deployment)
                # yield copy.deepcopy(deployment)
                # print(counters)
                all_deployment.append(copy.deepcopy(deployment))

                # gpu_group = []
                # for group in counters:
                #     print(group)
                #     for k, v in group:
                #         print(k,v)

                all_gpu_group.append([[ k + (v, ) for k, v in group.items()]  for group in counters])
                
                # print(counters)
                # exit()

            if seq_sum == (NUM_GPU-1) * NUM_STAGE:
                break
        # assert False
        # yield None

    deployment_generator(0,0,[Counter() for _ in range(NUM_GPU)])

    return all_deployment, all_gpu_group

def group_by_gpu(d):
    # gpu_group = [[] for _ in range(NUM_GPU)]

    # data = []

    # for m in range(NUM_MODEL):
    #     for r in range(NUM_REPLICATION):
    #         for s in range(NUM_STAGE):
    #             data.append([m, r, d[m][r][s]])

    # df = pd.DataFrame(data, columns=['m', 'r', 'g'])

    # df_group = df.groupby(['m', 'r', 'g']).size().to_frame('size').reset_index()
    # group_g = df_group.groupby(['g'])

    # # print(df_group)
    # # print(gpu_group)

    # for g_name, g_value in group_g:
    #     # print(type(df_group[['m', 'r', 'size']].to_records(index=False)))
    #     gpu_group[g_name] =  list(g_value[['m', 'r', 'size']].to_records(index=False))
    # exit()

    gpu_group = []
    for g in range(NUM_GPU):
        g_d = np.argwhere(d == g)
        # print ("g_d", g_d)
        m_set = set([e[0] for e in g_d])

        subgroup = []
        for m in m_set:
            for r in range(NUM_REPLICATION):
                stage_count = np.sum((g_d[:, 0] == m) & (g_d[:, 1] == r))
                subgroup.append((m, r, np.sum(stage_count)))    

        gpu_group.append(subgroup)    
    return gpu_group

def calculate_profit(d, cost_model, gpu_group, gpu_overhead):
    total_tp = 0

    for g in range(NUM_GPU):
        subgroup = gpu_group[g]
        total_mem = np.sum([model_mem[e[0]] / NUM_STAGE * e[2] for e in subgroup])
        if total_mem > gpu_mem[g]:
            return None

    for m in range(NUM_MODEL):
        for r in range(NUM_REPLICATION):
            if d[m][r][0] == -1: continue
            max_latency = 0
            appear_set = set(d[m][r])
            for s in range(NUM_STAGE):
                g_id = d[m][r][s]
                subgroup = gpu_group[g_id]
                stage_count = 0
                for e in subgroup:
                    model, replica, stage = e
                    if r == replica and model == m: 
                        stage_count = stage
                        break
                latency = cost_model.latency[m](ratio[m] * 64) / NUM_STAGE * stage_count  + gpu_overhead[g_id]
                max_latency = max(max_latency, latency)
            latency = max_latency + len(appear_set) * com_latency[m]
            # print("latency", latency, m, r)
            if latency > LATENCY_BOUND: return None

            total_tp += ratio[m] / latency

    return total_tp

def search_deployment(ratio):
    cost_model = CostModel("data")
    cost_model.fit()

    profits = []
    deployment, all_gpu_groups = get_all_deployment()
    print("all choices", len(deployment))
    for i, d in enumerate(tqdm(deployment)):
        d = np.array(d)
        gpu_group = all_gpu_groups[i]
        # gpu_group = group_by_gpu(d)
        # print("gpu_group", gpu_group)
        gpu_overhead = [cost_model.predict(gpu_group[g], ratio) for g in range(NUM_GPU)]

        profit = calculate_profit(d, cost_model, gpu_group, gpu_overhead)

        if profit is not None:
            profits.append((profit, d))

    profits.sort(key=lambda x: x[0], reverse=True)
    # print("profits", profits)

    return profits[0]

best = search_deployment(ratio)
print("best", best)
exit()

# search for all
solution_space = []

TOTAL_MEM = sum(gpu_mem)

def pick_solution(i, solution):
    if i >= len(model_mem):
        return
    used_mem = sum([s[i] * model_mem[i]  for i, s in enumerate(solution)])
    for r in range(1, int((TOTAL_MEM - used_mem) / model_mem[i])+1):
        for s in range(1, NUM_STAGE+1):
            next_solution = solution + [(ratio[i], r, s)]
            if i < len(model_mem):
                pick_solution(i + 1, next_solution)
            if i == len(model_mem) - 1:
                model_tp = []
                latency = 0
                flag = True
                for k, choice in enumerate(next_solution):
                    # print(next_solution)
                    rho, replica, stage = choice
                    tp = (
                        replica
                        * stage
                        * rho
                        / (inf_latency[k] + (stage - 1) * com_latency[k])
                    )
                    latency += (inf_latency[k] + (stage - 1) * com_latency[k]) * rho
                    flag &= (inf_latency[k] + (stage - 1) * com_latency[k]) < inf_latency[k] * 1.5
                    model_tp.append(tp)
                if latency <= LATENCY_BOUND and flag:
                    solution_space.append((min(model_tp), solution))
                    # print(len(solution_space))

# def pick_solution(i, solution):
#     if i >= len(model_mem):
#         return
#     for s in range(1, NUM_STAGE+1):
#         next_solution = solution + [(ratio[i], s)]
#         print(next_solution)
#         if i < len(model_mem):
#             pick_solution(i + 1, next_solution)
#         if i == len(model_mem) - 1:
#             model_tp = []
#             latency = 0
#             flag = True
#             for k, choice in enumerate(next_solution):
#                 rho, stage = choice
#                 tp = (
#                     stage
#                     * rho
#                     / (inf_latency[k] + (stage - 1) * com_latency[k])
#                 ) * 1000
#                 latency += (inf_latency[k] + (stage - 1) * com_latency[k]) * rho
#                 flag &= (inf_latency[k] + (stage - 1) * com_latency[k]) < inf_latency[k] * 1.5
#                 model_tp.append(tp)
#             print(model_tp, np.array(model_tp) / np.array(ratio))
#             lcm = np.lcm.reduce(np.ceil(np.array(model_tp) / np.array(ratio)).astype(int))
#             replicas = [lcm / tp for tp in model_tp]
#             mem = [model_mem[k] * r for k, r in enumerate(replicas)]
#             replicas = np.ceil(np.array(replicas) / (np.sum(mem) / TOTAL_MEM)).astype(int).tolist()
#             print(lcm, replicas, mem)

#             if latency <= LATENCY_BOUND and flag:
#                 solution = [s + (replicas[k], ) for k, s in enumerate(solution)]
#                 solution_space.append((min(model_tp), solution))
#                 print(len(solution_space))


pick_solution(0, [])
solution_space.sort(key=lambda x: x[0], reverse=True)
print("solution_space", len(solution_space), solution_space[:10])




import copy
from pulp import *
import numpy as np
from tqdm import tqdm

from hfutils.logger import Logger

logger = Logger(__file__, "info", 500000, 2)

NUM_GPUS = 8
NUM_STAGES = NUM_GPUS
NUM_REPLICA = 4
NUM_MODEL = 4

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)
solver = getSolver('GUROBI_CMD', msg=False)

MAX_POWER = 250
MIN_POWER = 50
GPU_MEMORY = 25000

data_prop = [0.49, 0.01, 0.01, 0.49]
data_prop = np.cumsum(data_prop[::-1])[::-1]
latency = np.array([35, 55, 88, 125])
utilization = [0.086, 0.11, 0.12, 0.15]
memory = [300, 1200, 6000, 13000]

time_ratio = data_prop * latency
# time_ratio = np.round(time_ratio * 1000)
time_ratio = time_ratio / np.min(time_ratio)
time_ratio = np.array([int(np.round(x)) for x in time_ratio])

logger.info("base replication ratio %s", time_ratio)

def problem_formulate(settings):
    # print(settings)
    choices = [
        LpVariable.dicts(f"{m}-placement", (range(settings[m][0]), range(settings[m][1]), range(NUM_GPUS)), lowBound=0, upBound=1, cat="Integer")
        for m in range(NUM_MODEL)
    ]
    prob = LpProblem("Carbon_Problem", LpMinimize)
    Z = LpVariable("Z", lowBound=0)

    # Objective
    prob += Z

    avg_latency = sum([ 
        1 / settings[m][1] * data_prop[m] * (latency[m] / settings[m][0] + (settings[m][0] - 1) * 4) * MIN_POWER
        for m in range(NUM_MODEL)
    ])
    for g in range(NUM_GPUS):

        # Minmax equivalence
        prob += Z >= (1-lpSum([ choices[m][s][r][g] * utilization[m]
            for m in range(NUM_MODEL)
            for s in range(settings[m][0])
            for r in range(settings[m][1])
        ])) * avg_latency

        # Utilization constraint
        prob += lpSum([ choices[m][s][r][g] * utilization[m]
            for m in range(NUM_MODEL)
            for s in range(settings[m][0])
            for r in range(settings[m][1])
        ]) <= 1

        # Memory constraint
        prob += GPU_MEMORY >= lpSum([ 
            choices[m][s][r][g] * memory[m] / settings[m][0]
            for m in range(NUM_MODEL)
            for s in range(settings[m][0])
            for r in range(settings[m][1])
        ])

    for m in range(NUM_MODEL):
        for s in range(settings[m][0]):
            for r in range(settings[m][1]):
                prob += lpSum([ choices[m][s][r][g] for g in range(NUM_GPUS) ]) == 1

    prob.solve(solver)

    # print("status", prob.status)
    if prob.status < 0:
        logger.error("status %s", prob.status)
        return None

    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)

    return value(prob.objective), prob

min_objective = 10000000000
min_prob = None
solution_count = 0

import itertools
import random

all_stage_permutations = list(itertools.product([x+1 for x in range(NUM_STAGES)], repeat=NUM_MODEL))
all_replica_permutations = list(itertools.product([x+1 for x in range(NUM_REPLICA)], repeat=NUM_MODEL))

random.shuffle(all_stage_permutations)
random.shuffle(all_replica_permutations)

print(len(all_stage_permutations), len(all_replica_permutations))

# pbar = tqdm(total=len(all_stage_permutations) * len(all_replica_permutations))
pbar = tqdm(total=len(all_stage_permutations) * NUM_REPLICA)
for s_perm in all_stage_permutations:
    overhead = (np.array(s_perm) - 1) * 4 > latency * 0.1
    if np.any(overhead):
        pbar.update(NUM_REPLICA)
        continue   

    for r in range(NUM_REPLICA):
        r_perm = tuple((time_ratio * (r+1)).tolist())
    # for r_perm in all_replica_permutations:
        settings = list(zip(s_perm, r_perm))

        pbar.update(1)

        if sum([setting[1] * memory[k] for k, setting in enumerate(settings)]) > GPU_MEMORY * NUM_GPUS:
            logger.info("no feasilble memory %s", settings)
            continue

        solution = problem_formulate(settings)
        if solution is None:
            logger.info("no feasilble solution %s", settings)
            continue
        objective, prob = solution
        if objective is None: 
            logger.info("no feasilble solution %s", settings)
            continue

        if objective < min_objective:
            min_objective = objective
            min_prob = prob
            logger.info("new_solution %s, %s", min_objective, settings)
            solution_dict = {
                v.name: v.varValue for v in prob.variables()
            }
            with open(os.path.join(os.path.dirname(__file__), "solution.json"), "w") as fp:
                json.dump(solution_dict, fp)

# def dfs(i, settings):

#     global min_value
#     global min_prob
#     global solution_count

#     if i >= NUM_MODEL: return
#     for s in range(NUM_STAGES):
#         for r in range(NUM_REPLICA):
#             new_settings = settings + [(s+1,r+1)]
#             dfs(i+1, copy.deepcopy(new_settings))

#             if i == NUM_MODEL - 1:
#                 assert len(new_settings) == NUM_MODEL
#                 # total_instance = sum([s*r for s, r in new_settings])
#                 # if total_instance < NUM_GPUS: continue
#                 if sum([setting[1] * memory[k] for k, setting in enumerate(new_settings)]) > GPU_MEMORY * NUM_GPUS:
#                     logger.info("no feasilble memory %s", new_settings)
#                     continue
#                 solution = problem_formulate(new_settings)
#                 if solution is None:
#                     logger.info("no feasilble solution %s", new_settings)
#                     continue
#                 value, prob = solution
#                 if value is None: 
#                     logger.info("no feasilble solution %s", new_settings)
#                     continue

#                 # print(value)

#                 if value < min_value:
#                     min_value = value
#                     min_prob = prob
#                     logger.info("new_solution %s, %s", min_value, new_settings)
#                     solution_dict = {
#                       v.name: v.varValue for v in prob.variables()
#                     }
#                     with open(os.path.join(os.path.dirname(__file__), "solution.json"), "w") as fp:
#                         json.dump(solution_dict, fp)
                    
#                     # for v in prob.variables():
#                     #     logger.info("%s = %s", v.name, v.varValue)

#                 pbar.update(1)

# dfs(0, [])
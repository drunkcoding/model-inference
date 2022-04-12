from dataclasses import dataclass, field
import itertools, random
import json
from pulp import *
import pulp
import numpy as np
import os
import logging
from hfutils.logger import Logger
from tqdm import tqdm
from functools import partial
from transformers import HfArgumentParser

logger = Logger(__file__, logging.INFO, 500000, 2)

# os.environ['GUROBI_HOME'] = "/home/oai/gurobi951/linux64"

# export GUROBI_HOME="${HOME}/gurobi951/linux64"
# export PATH="${PATH}:${GUROBI_HOME}/bin"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"


def func(x, b, c):
    return np.power(x, 2) / (np.power(x, 2) + b * x + c)


with open("tests/confidence/confidence.json", "r") as fp:
    prop_config = json.load(fp)

with open("tests/profile/nsys_extract.json", "r") as fp:
    kernel_config = json.load(fp)

with open("tests/profile/profile.json", "r") as fp:
    transmit_config = json.load(fp)

with open("tests/kernel_duration/memory_params.json", "r") as fp:
    memory_config = json.load(fp)

with open("tests/kernel_duration/latency.json", "r") as fp:
    latency_config = json.load(fp)

with open("tests/kernel_duration/hidden_size.json", "r") as fp:
    hidden_config = json.load(fp)

with open("repository/repo_profile/profile.json", "r") as fp:
    transmit_config = json.load(fp)


@dataclass
class Arguments:
    model_type: str = field(metadata={"help": "Model type choose from t5,bert,vit,gpt"})
    num_gpus: int = field(metadata={"help": "Number of GPU in the Cluster"})


parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

GPU_MEMORY = 25000

BATCHES = [1, 2, 4, 6, 8, 16, 32, 48, 64]
MODE_KEYS = list(prop_config[args.model_type].keys())

NUM_STAGES = 1
NUM_REPLICA = 8
NUM_MODEL = len(MODE_KEYS)
NUM_GPUS = args.num_gpus

# pulp.pulpTestAll()

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)
solver = getSolver("GUROBI_CMD")

data_prop = prop_config[args.model_type]
data_prop = np.cumsum(list(data_prop.values())[::-1])[::-1]
# latency = latency_config[args.model_type]
# kernel = np.poly1d(kernel_config[args.model_type])
# memory = memory_config[args.model_type]
hidden = hidden_config[args.model_type]

kernel = {
    i: partial(func, b=kernel_config[key][0], c=kernel_config[key][1])
    for i, key in enumerate(MODE_KEYS)
}
memory = {i: np.poly1d(memory_config[key]) for i, key in enumerate(MODE_KEYS)}
latency = {i: latency_config[key]["1"] for i, key in enumerate(MODE_KEYS)}

logger.info("%s kernel %s", args.model_type, kernel)
logger.info("%s memory %s", args.model_type, memory)
logger.info("%s latency %s", args.model_type, latency)

tutil_func = [
    [
        np.poly1d(transmit_config["numa"])
        if (g < 4 and gp < 4) or (g >= 4 and gp >= 4)
        else np.poly1d(transmit_config["cross"])
        for g in range(NUM_GPUS)
    ]
    for gp in range(NUM_GPUS)
]

for g in range(NUM_GPUS):
    tutil_func[g][g] = lambda x: 0


def problem_formulate(settings):

    # p_stages, p_replicas, p_batch = settings[m]
    print(settings)
    choices = [
        LpVariable.dicts(
            f"{m}-placement",
            (range(settings[m][0]), range(settings[m][1]), range(NUM_GPUS)),
            lowBound=0,
            upBound=1,
            cat="Integer",
        )
        for m in range(NUM_MODEL)
    ]

    prob = LpProblem("Placement Problem", LpMaximize)

    prob += lpSum(
        [
            choices[m][s][r][g] * kernel[m](settings[m][2])
            - lpSum(
                [
                    tutil_func[gp][g](settings[m][2] * hidden[MODE_KEYS[m]] * 8)
                    * (choices[m][s][r][g] + choices[m - 1][sp][rp][gp])
                    for sp in range(settings[m - 1][0])
                    for rp in range(settings[m - 1][1])
                    for gp in range(NUM_GPUS)
                ]
            )
            for m in range(1, NUM_MODEL)
            for s in range(settings[m][0])
            for r in range(settings[m][1])
            for g in range(NUM_GPUS)
        ]
    ) + lpSum(
        [
            choices[0][s][r][g] * kernel[0](settings[0][2])
            for s in range(settings[0][0])
            for r in range(settings[0][1])
            for g in range(NUM_GPUS)
        ]
    )

    for g in range(NUM_GPUS):
        # Memory constraint
        prob += GPU_MEMORY >= lpSum(
            [
                choices[m][s][r][g] * memory[m](settings[m][2]) / settings[m][0]
                for m in range(NUM_MODEL)
                for s in range(settings[m][0])
                for r in range(settings[m][1])
            ]
        )

        # Utilization constraint
        prob += (
            lpSum(
                [
                    choices[m][s][r][g] * kernel[m](settings[m][2])
                    for m in range(NUM_MODEL)
                    for s in range(settings[m][0])
                    for r in range(settings[m][1])
                ]
            )
            <= 1
        )

    # Only appear once
    for m in range(NUM_MODEL):
        for s in range(settings[m][0]):
            for r in range(settings[m][1]):
                prob += lpSum([choices[m][s][r][g] for g in range(NUM_GPUS)]) == 1

    prob.solve(solver)

    # print("status", prob.status)
    if prob.status < 0:
        logger.error("status %s", prob.status)
        return None

    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)

    return value(prob.objective), prob


max_objective = -10000000000
max_prob = None
solution_count = 0

all_stage_permutations = list(
    itertools.product([x + 1 for x in range(NUM_STAGES)], repeat=NUM_MODEL)
)
# all_replica_permutations = list(itertools.product([x+1 for x in range(NUM_REPLICA)], repeat=NUM_MODEL))
# MAX_BATCH_SIZE = 256

random.shuffle(all_stage_permutations)
# random.shuffle(all_replica_permutations)

print(len(all_stage_permutations))

pbar = tqdm(total=len(all_stage_permutations) * len(BATCHES) * NUM_REPLICA)
for s_perm in all_stage_permutations:
    stage_perm = np.array(s_perm)
    for bsz in BATCHES:
        stage_latency = np.array(list(latency.values())) / stage_perm
        intra_latency = (
            stage_perm
            * np.array(
                [
                    tutil_func[1][0](bsz * data_prop[i] * hidden[key] * 8)
                    for i, key in enumerate(MODE_KEYS)
                ]
            )
            * 1000
        )
        print("stage", stage_latency, intra_latency)
        time_ratio = data_prop * (stage_latency + intra_latency)
        print("time_ratio", time_ratio)
        time_ratio = time_ratio / np.min(time_ratio)
        
        time_ratio = np.array([int(np.round(x)) for x in time_ratio])
        logger.info("%s %s time_ratio scaled %s", args.model_type, args.num_gpus, time_ratio)

        for r in range(NUM_REPLICA):
            r_perm = tuple((time_ratio * (r + 1)).tolist())

            settings = list(zip(s_perm, r_perm, data_prop * bsz))

            pbar.update(1)

            if (
                sum(
                    [
                        setting[1] * memory[k](setting[2])
                        for k, setting in enumerate(settings)
                    ]
                )
                > GPU_MEMORY * NUM_GPUS
            ):
                logger.info("%s %s no feasilble memory %s", args.model_type, args.num_gpus, settings)
                continue

            solution = problem_formulate(settings)
            if solution is None:
                logger.info("%s %s no feasilble solution %s", args.model_type, args.num_gpus, settings)
                continue
            objective, prob = solution
            if objective is None:
                logger.info("%s %s no feasilble solution %s", args.model_type, args.num_gpus, settings)
                continue

            if objective > max_objective:
                max_objective = objective
                max_prob = prob
                logger.info("%s %s new_solution %s, %s", args.model_type, args.num_gpus, max_objective, settings)
                solution_dict = {v.name: v.varValue for v in prob.variables()}
                with open(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"{args.model_type}-{args.num_gpus}-solution.json",
                    ),
                    "w",
                ) as fp:
                    json.dump(solution_dict, fp)

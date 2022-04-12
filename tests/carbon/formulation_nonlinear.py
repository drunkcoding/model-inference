from dataclasses import dataclass, field
import itertools, random
import json
from pyomo.environ import *
from pyomo.environ import value
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
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

with open("tests/profile/kernel.json", "r") as fp:
    kernel_measure = json.load(fp)

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

# BATCHES = [1, 2, 4, 6, 8, 16, 32]
BATCHES = [4]
MODEL_KEYS = list(prop_config[args.model_type].keys())

NUM_STAGES = 1
NUM_REPLICA = args.num_gpus
NUM_MODEL = len(MODEL_KEYS)
NUM_GPUS = args.num_gpus

data_prop = prop_config[args.model_type]
data_prop = np.cumsum(list(data_prop.values())[::-1])[::-1]
# latency = latency_config[args.model_type]
# kernel = np.poly1d(kernel_config[args.model_type])
# memory = memory_config[args.model_type]
hidden = hidden_config[args.model_type]

kernel = {
    i: partial(func, b=kernel_config[key][0], c=kernel_config[key][1])
    for i, key in enumerate(MODEL_KEYS)
}
memory = {i: np.poly1d(memory_config[key]) for i, key in enumerate(MODEL_KEYS)}
latency = {i: latency_config[key]["1"] for i, key in enumerate(MODEL_KEYS)}

print(memory)

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

    model = ConcreteModel()
    model.constraints = ConstraintList()

    # model.choices = [Var(
    #     range(settings[m][0]),
    #     range(settings[m][1]),
    #     range(NUM_GPUS),
    #     bounds=(0, 1),
    #     domain=Integers,
    #     initialize=0,
    # ) for m in range(NUM_MODEL)]
    # for m in range(NUM_MODEL):
    #     model.choices[m].construct()
    print(settings)
    model.choices = Var(
        range(NUM_MODEL),
        range(NUM_STAGES),
        range(NUM_REPLICA * max([setting[1] for setting in settings])),
        range(NUM_GPUS),
        bounds=(0, 1),
        domain=Integers,
        initialize=0,
    )
    model.choices.construct()

    # kernel_func = (
    #     lambda m, bsz: kernel[m](settings[m][2])
    #     if not bsz in BATCHES
    #     else kernel_measure[MODEL_KEYS[m]][str(bsz)]
    # )

    kernel_func = lambda m, bsz: kernel_measure[MODEL_KEYS[m]][str(bsz)]

    objective_rule = sum(
        [
            model.choices[m, s, r, g] * kernel_func(m, settings[m][2])
            - sum(
                [
                    tutil_func[gp][g](settings[m][2] * hidden[MODEL_KEYS[m]] * 8)
                    * model.choices[m, s, r, g]
                    * model.choices[m - 1, sp, rp, gp]
                    for sp in range(settings[m - 1][0])
                    for rp in range(settings[m - 1][1])
                    for gp in range(NUM_GPUS)
                ]
            )
            / settings[m - 1][1]
            for m in range(1, NUM_MODEL)
            for s in range(settings[m][0])
            for r in range(settings[m][1])
            for g in range(NUM_GPUS)
        ]
    ) + sum(
        [
            model.choices[0, s, r, g] * kernel_func(0, settings[0][2])
            for s in range(settings[0][0])
            for r in range(settings[0][1])
            for g in range(NUM_GPUS)
        ]
    )
    # - sum([model.choices[m, s, r, g]
    #     for m in range(NUM_MODEL)
    #     for s in range(NUM_STAGES)
    #     for r in range(NUM_REPLICA)
    #     for g in range(NUM_GPUS)])
    model.obj = Objective(expr=objective_rule, sense=maximize, name="utilization")

    for g in range(NUM_GPUS):
        # Memory constraint
        model.constraints.add(
            sum(
                [
                    model.choices[m, s, r, g]
                    * memory[m](settings[m][2])
                    / settings[m][0]
                    for m in range(NUM_MODEL)
                    for s in range(settings[m][0])
                    for r in range(settings[m][1])
                ]
            )
            <= GPU_MEMORY
        )

        # Utilization constraint
        model.constraints.add(
            sum(
                [
                    model.choices[m, s, r, g] * kernel_func(m, settings[m][2])
                    for m in range(NUM_MODEL)
                    for s in range(settings[m][0])
                    for r in range(settings[m][1])
                ]
            )
            <= 1
        )

        model.constraints.add(
            sum(
                [
                    model.choices[m, s, r, g]
                    for m in range(NUM_MODEL)
                    for s in range(settings[m][0])
                    for r in range(settings[m][1])
                ]
            )
            >= 1
        )

    # Only appear once
    for m in range(NUM_MODEL):
        for s in range(settings[m][0]):
            for r in range(settings[m][1]):
                model.constraints.add(
                    sum([model.choices[m, s, r, g] for g in range(NUM_GPUS)]) == 1
                )

    opt = SolverFactory("gurobi")
    results = opt.solve(model)
    # results.write()

    results = opt.solve(model)  # Solving a model instance
    # model.load(results) # Loading solution into results object

    if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal
    ):
        # Do something when the solution in optimal and feasible
        solution = []
        for v in model.component_objects(Var, active=True):
            for index in v:
                # print(index, v[index].value)
                solution.append((index, v[index].value))
        return value(model.obj), solution
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # Do something when model in infeasible
        return
    else:
        # Something else is wrong
        print("Solver Status: ", results.solver.status)

    # print(model.choices)

    # print(value(model.obj))
    # print(value(model.choices))

    # for v in model.component_objects(Var, active=True):
    #     print ("Variable component object",v)
    #     for index in v:
    #         print ("   ", index, v[index].value)


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
                    for i, key in enumerate(MODEL_KEYS)
                ]
            )
            * 1000
        )
        print("stage", stage_latency, intra_latency)
        time_ratio = data_prop * (stage_latency + intra_latency)
        # new_list = set(time_ratio)
        # new_list.remove(min(new_list))
        time_ratio /= min(time_ratio)
        time_ratio = np.array([int(np.ceil(x)) for x in time_ratio])

        # time_ratio = time_ratio / np.min(time_ratio)

        logger.info(
            "%s %s time_ratio scaled %s", args.model_type, args.num_gpus, time_ratio
        )

        batch = data_prop * bsz
        batch = np.array([int(np.ceil(x)) for x in batch])

        for i, b in enumerate(batch):
            gap = 100000
            b_arr = np.abs(np.array(BATCHES) - b)
            batch[i] = BATCHES[np.argmin(b_arr)]

        print(data_prop)
        # batch = batch / batch[0] * bsz

        # batch = np.array([bsz] * len(data_prop))

        for r in range(NUM_REPLICA):
            r_perm = (time_ratio * (r + 1)).tolist()

            # new_list = set(r_perm)
            # new_list.remove(min(new_list))

            # r_perm /= min(new_list)
            # r_perm = [int(np.ceil(x)) for x in r_perm]

            settings = list(zip(s_perm, r_perm, batch))

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
                logger.info(
                    "%s %s no feasilble memory %s",
                    args.model_type,
                    args.num_gpus,
                    settings,
                )
                continue

            solution = problem_formulate(settings)
            if solution is None:
                logger.info(
                    "%s %s no feasilble solution %s",
                    args.model_type,
                    args.num_gpus,
                    settings,
                )
                continue
            objective, prob = solution
            if objective is None:
                logger.info(
                    "%s %s no feasilble solution %s",
                    args.model_type,
                    args.num_gpus,
                    settings,
                )
                continue

            if objective > max_objective:
                max_objective = objective
                max_prob = prob
                logger.info(
                    "%s %s new_solution %s, %s",
                    args.model_type,
                    args.num_gpus,
                    max_objective,
                    settings,
                )
                # solution_dict = {v.name: v.varValue for v in prob.variables()}
                solution_dict = {MODEL_KEYS[index[0]]: list() for index, _ in prob}
                for index, v in prob:
                    if v == 1.0:
                        solution_dict[MODEL_KEYS[index[0]]].append(index)
                solution_dict["batch_size"] = batch.tolist()
                with open(
                    os.path.join(
                        os.path.dirname(__file__),
                        f"{args.model_type}-{args.num_gpus}-solution.json",
                    ),
                    "w",
                ) as fp:
                    json.dump(solution_dict, fp)

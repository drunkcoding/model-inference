import copy
from pulp import *
import numpy as np

NUM_GPUS = 8
NUM_STAGES = NUM_GPUS
NUM_REPLICA = NUM_GPUS
NUM_MODEL = 4

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)
solver = getSolver('GUROBI_CMD', msg=True)

MAX_POWER = 250
MIN_POWER = 50
GPU_MEMORY = 25000

data_prop = [0.49, 0.01, 0.01, 0.49]
data_prop = np.cumsum(data_prop[::-1])[::-1]
latency = [35, 55, 88, 93]
utilization = [0.086, 0.11, 0.12, 0.15]
memory = [300, 1200, 6000, 13000]

latency_coef = [
    [1/s/r for s in range(1, NUM_STAGES+1)]
    for r in range(1, NUM_REPLICA+1)
] 

choices = [
    LpVariable.dicts(f"{m}-placement", (range(NUM_STAGES), range(NUM_REPLICA), range(NUM_GPUS)), lowBound=0, upBound=1, cat="Integer")
    for m in range(NUM_MODEL)
]

prob = LpProblem("Carbon_Problem", LpMinimize)
Z = LpVariable("Z", lowBound=0)
stages = [
    LpVariable(f"{m}-stage", lowBound=1, upBound=NUM_STAGES, cat="Integer")
    for m in range(NUM_MODEL)
]
replications = [
    LpVariable(f"{m}-replica", lowBound=1, upBound=NUM_REPLICA, cat="Integer")
    for m in range(NUM_MODEL)
]

prob += Z + lpSum([ 
    (stages[m] + 1) / stages[m] / replications[m] * data_prop[m] 
    * (latency[m] + (stages[m] - 1) * 4) * MAX_POWER * utilization[m]
    for m in range(NUM_MODEL)
])

for g in range(NUM_GPUS):
    # Memory constraint
    prob += GPU_MEMORY >= lpSum([ 
        choices[m][s][r][g] * memory[m] / stages[m]
        for m in range(NUM_MODEL)
        for s in range(NUM_STAGES)
        for r in range(NUM_REPLICA)
    ])

    # Utilization constraint
    prob += lpSum([ 
        choices[m][s][r][g] * utilization[m]
        for m in range(NUM_MODEL)
        for s in range(NUM_STAGES)
        for r in range(NUM_REPLICA)
    ]) <= 1

    # Minmax equivalence
    prob += Z >= (1 - lpSum([ 
        choices[m][s][r][g] * utilization[m]
        for m in range(NUM_MODEL)
        for s in range(NUM_STAGES)
        for r in range(NUM_REPLICA)
    ])) * lpSum([ 
        (stages[m] + 1) / stages[m] / replications[m] * data_prop[m] * (latency[m] + (stages[m] - 1) * 4) * MIN_POWER
        for m in range(NUM_MODEL)
    ])

for m in range(NUM_MODEL):
    # Number of stages 
    for r in range(NUM_REPLICA):
        prob += lpSum([ 
            choices[m][s][r][g]
            for g in range(NUM_GPUS)
            for s in range(NUM_STAGES)
            for r in range(NUM_REPLICA)
        ]) == stages[m]

    # Number of replications 
    for s in range(NUM_STAGES):
        prob += lpSum([ 
            choices[m][s][r][g]
            for g in range(NUM_GPUS)
            for s in range(NUM_STAGES)
            for r in range(NUM_REPLICA)
        ]) == replications[m]

prob.writeLP(os.path.join("tests", "carbon", f"full_search.lp"))
prob.solve(solver)
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
from pulp import *
import numpy as np
import os

# os.environ['GUROBI_HOME'] = "/home/oai/gurobi951/linux64"

# export GUROBI_HOME="${HOME}/gurobi951/linux64"
# export PATH="${PATH}:${GUROBI_HOME}/bin"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

NUM_STAGES = range(8)
NUM_REPLICA = range(2)
NUM_MODEL = range(4)
NUM_GPUS = range(8)

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)
solver = getSolver('GUROBI_CMD')

data_prop = [0.49, 0.01, 0.01, 0.49]
data_prop = np.cumsum(data_prop[::-1])[::-1]
latency = [35, 55, 88, 93]
utilization = [0.086, 0.11, 0.12, 0.15]

utilization = makeDict([[x for x in NUM_GPUS]], utilization, 0)
print(utilization)
idle_power = 20

prob = LpProblem("Carbon Problem", LpMaximize)
Z = LpVariable("Z")
choices = LpVariable.dicts("Choice", (NUM_MODEL, NUM_STAGES, NUM_REPLICA), lowBound=0, upBound=7, cat="Integer")

prob += Z
for g in NUM_GPUS:
    prob += Z <= lpSum([ choices[m][s][r] * utilization[m]
        for m in NUM_MODEL
        for s in NUM_STAGES
        for r in NUM_REPLICA
        if choices[m][s][r] == g
    ])
    prob += lpSum([ choices[m][s][r] * utilization[m]
        for m in NUM_MODEL
        for s in NUM_STAGES
        for r in NUM_REPLICA
         if choices[m][s][r] == g
    ]) <= 1

# for m in NUM_MODEL:
#     for s in NUM_STAGES:
#         for r in NUM_REPLICA:
#             prob += lpSum([ choices[m][s][r][g] for g in NUM_GPUS ]) == 1
#             # for g in NUM_GPUS:
#             #     prob += choices[m][s][r][g] <= 1
#             #     prob += choices[m][s][r][g] >= 0

# The problem data is written to an .lp file
prob.writeLP(os.path.join("tests", "carbon", "solver.lp"))

# The problem is solved using PuLP's choice of Solver
prob.solve(solver)

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

if prob.status < 0:
    exit()

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

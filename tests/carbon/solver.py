from pulp import *
import numpy as np

solver_list = listSolvers(onlyAvailable=True)
print(solver_list)
solver = getSolver('GUROBI_CMD')

NUM_STAGES = range(8)
NUM_REPLICA = range(1)
NUM_MODEL = range(4)
NUM_GPUS = range(8)

data_prop = [0.49, 0.01, 0.01, 0.49]
data_prop = np.cumsum(data_prop[::-1])[::-1]
latency = [35, 55, 88, 93]
utilization = [0.086, 0.11, 0.12, 0.15]

utilization = makeDict([[x for x in NUM_GPUS]], utilization, 0)
print(utilization)
idle_power = 20

prob = LpProblem("Carbon Problem", LpMaximize)
Z = LpVariable("Z")
choices = LpVariable.dicts("Choice", (NUM_MODEL, NUM_STAGES, NUM_REPLICA, NUM_GPUS), lowBound=0, upBound=1, cat="Integer")

prob += Z
for g in NUM_GPUS:
    prob += Z <= lpSum([ choices[m][s][r][g] * utilization[m]
        for m in NUM_MODEL
        for s in NUM_STAGES
        for r in NUM_REPLICA
    ])
    prob += lpSum([ choices[m][s][r][g] * utilization[m]
        for m in NUM_MODEL
        for s in NUM_STAGES
        for r in NUM_REPLICA
    ]) <= 1

for m in NUM_MODEL:
    for s in NUM_STAGES:
        for r in NUM_REPLICA:
            prob += lpSum([ choices[m][s][r][g] for g in NUM_GPUS ]) == 1
            # for g in NUM_GPUS:
            #     prob += choices[m][s][r][g] <= 1
            #     prob += choices[m][s][r][g] >= 0

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

# The optimised objective function value is printed to the screen
# print("Total Cost of Ingredients per can = ", value(prob.objective))
# print(prob.constraints)
# # The optimised objective function value is printed to the screen
# print("Total Cost of Ingredients per can = ", value(prob.objective))

# # Create the 'prob' variable to contain the problem data
# prob = LpProblem("The Whiskas Problem", LpMinimize)


# # The 2 variables Beef and Chicken are created with a lower limit of zero
# x1 = LpVariable("ChickenPercent", 0, None, LpInteger)
# x2 = LpVariable("BeefPercent", 0)

# # The objective function is added to 'prob' first
# prob += 0.013 * x1 + 0.008 * x2, "Total Cost of Ingredients per can"


# # The five constraints are entered
# prob += x1 + x2 == 100, "PercentagesSum"
# prob += 0.100 * x1 + 0.200 * x2 >= 8.0, "ProteinRequirement"
# prob += 0.080 * x1 + 0.100 * x2 >= 6.0, "FatRequirement"
# prob += 0.001 * x1 + 0.005 * x2 <= 2.0, "FibreRequirement"
# prob += 0.002 * x1 + 0.005 * x2 <= 0.4, "SaltRequirement"

# # The problem data is written to an .lp file
# prob.writeLP("WhiskasModel.lp")

# # The problem is solved using PuLP's choice of Solver
# prob.solve()

# # The status of the solution is printed to the screen
# print("Status:", LpStatus[prob.status])

# # Each of the variables is printed with it's resolved optimum value
# for v in prob.variables():
#     print(v.name, "=", v.varValue)


# # The optimised objective function value is printed to the screen
# print("Total Cost of Ingredients per can = ", value(prob.objective))


# NUM_STAGES = 8
# NUM_REPLICA = 8
# NUM_MODEL = 4
# NUM_GPUS = 8

# from random import choices
# from gekko import GEKKO
# solver = GEKKO(remote=False)
# solver.options.SOLVER = 3

# choices = solver.Array(solver.Var, (NUM_MODEL, NUM_STAGES, NUM_REPLICA, NUM_GPUS), integer=True, lb=0, ub=1, value=0)
# Z = solver.Var(lb=0)

# solver.Maximize(Z)

# # print(choices[0][0][0].value)

# for m in range(NUM_MODEL):
#     for s in range(NUM_STAGES):
#         for r in range(NUM_REPLICA):
#             solver.Equation(solver.sum([ choices[m][s][r][g] for g in range(NUM_GPUS) ]) == 1)

# for g in range(NUM_GPUS):
#     solver.Equations([
#         Z <= solver.sum([choices[m][s][r][g] * utilization[m] for m in range(NUM_MODEL) for s in range(NUM_STAGES) for r in range(NUM_REPLICA) ]), 
#         solver.sum([choices[m][s][r][g] * utilization[m] for m in range(NUM_MODEL) for s in range(NUM_STAGES) for r in range(NUM_REPLICA) ]) <= 1
#     ])

# print("starting")

# solver.solve(debug=100)
# print('choices: ', choices.value)
# print('Z:  ',Z.value)
import os
from random import randrange
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from pyparsing import nums

NUM_GPUS = 8
NUM_STAGES = NUM_GPUS
NUM_REPLICA = NUM_GPUS
NUM_MODEL = 4

MAX_POWER = 250
MIN_POWER = 50
GPU_MEMORY = 25000

data_prop = [0.49, 0.01, 0.01, 0.49]
data_prop = np.cumsum(data_prop[::-1])[::-1]
latency = [35, 55, 88, 93]
utilization = [0.086, 0.11, 0.12, 0.15]
memory = [300, 1200, 6000, 13000]

model = ConcreteModel()
model.constraints = ConstraintList()

# model.choices = [Var(
#     range(NUM_STAGES), range(NUM_REPLICA), range(NUM_GPUS), 
#     domain=Integers, bounds=(0,1), doc=f"{m}-placement"
# ) for m in range(NUM_MODEL)]

# model.choices = [
#     [
#         [
#             [Var(bounds=(0, 1), domain=Integers, initialize=0) for g in range(NUM_GPUS)]
#             for r in range(NUM_REPLICA)
#         ]
#         for s in range(NUM_STAGES)
#     ]
#     for m in range(NUM_MODEL)
# ]
model.choices = Var(range(NUM_MODEL), range(NUM_STAGES), range(NUM_REPLICA), range(NUM_GPUS), bounds=(0, 1), domain=Integers, initialize=0)
model.choices.construct()

model.Z = Var(bounds=(0.0, None), doc="Z", initialize=0)
model.Z.construct()
model.stages = Var(range(NUM_MODEL), doc=f"stage", bounds=(1, NUM_STAGES), domain=Integers, initialize=1)
model.stages.construct()
model.replications = Var(range(NUM_MODEL), doc=f"replica", bounds=(1, NUM_REPLICA), domain=Integers, initialize=1)
model.replications.construct()

objective_rule = model.Z + sum([ 
    (model.stages[m] + 1) / model.stages[m] / model.replications[m] * data_prop[m] 
    * (latency[m] + (model.stages[m] - 1) * 4) * MAX_POWER * utilization[m]
    for m in range(NUM_MODEL)
])
model.obj = Objective(expr = objective_rule, sense=minimize)

for g in range(NUM_GPUS):
    # Memory constraint
    model.constraints.add(
        GPU_MEMORY >= sum([ 
            model.choices[m,s,r,g] * memory[m] / model.stages[m]
            for m in range(NUM_MODEL)
            for s in range(NUM_STAGES)
            for r in range(NUM_REPLICA)
        ])
    )

    # Utilization constraint
    model.constraints.add( 
        sum([ 
            model.choices[m,s,r,g] * utilization[m]
            for m in range(NUM_MODEL)
            for s in range(NUM_STAGES)
            for r in range(NUM_REPLICA)
        ]) <= 1
    )

    # Minmax equivalence
    model.constraints.add( 
        model.Z >= (1 - sum([ 
            model.choices[m,s,r,g] * utilization[m]
            for m in range(NUM_MODEL)
            for s in range(NUM_STAGES)
            for r in range(NUM_REPLICA)
        ])) * sum([ 
            (model.stages[m] + 1) / model.stages[m] / model.replications[m] * data_prop[m] * (latency[m] + (model.stages[m] - 1) * 4) * MIN_POWER
            for m in range(NUM_MODEL)
        ])
    )

for m in range(NUM_MODEL):
    # Number of stages
    for r in range(NUM_REPLICA):
        model.constraints.add(
            sum([ 
                model.choices[m,s,r,g]
                for g in range(NUM_GPUS)
                for s in range(NUM_STAGES)
            ]) == model.stages[m]
        )

    # Number of replications
    for s in range(NUM_STAGES):
        model.constraints.add(
            sum([ 
                model.choices[m,s,r,g]
                for g in range(NUM_GPUS)
                for r in range(NUM_REPLICA)
            ]) == model.replications[m]
        )

if __name__ == '__main__':
    # instance = model.create_instance()
    opt = SolverFactory("gurobi")
    results = opt.solve(model)
    model.pprint()
    results.write()

    for v in model.component_objects(Var, active=True):
        print ("Variable component object",v)
        for index in v:
            print ("   ", index, v[index].value)
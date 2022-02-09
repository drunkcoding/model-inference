import numpy as np
from operator import itemgetter

NUM_GPU = 8
NUM_STAGE = 8
NUM_REPLICATION = 20
LATENCY_BOUND = 150


ratio = [0.8, 0.1, 0.05, 0.05]
inf_latency = [28, 46, 88, 102]
com_latency = [3, 4, 8, 12]
model_mem = [267, 928, 2.92 * 1024, 10.3 * 1024]
gpu_mem = [25 * 1024] * NUM_GPU
num_replica = [20, 10, 5, 2]

# search for all
solution_space = []

TOTAL_MEM = sum(gpu_mem)

# def pick_solution(i, solution):
#     if i >= len(model_mem):
#         return
#     used_mem = sum([s[1] * model_mem[i]  for i, s in enumerate(solution)])
#     for r in range(1, int((TOTAL_MEM - used_mem) / model_mem[i])+1):
#         for s in range(1, NUM_STAGE+1):
#             next_solution = solution + [(ratio[i], r, s)]
#             if i < len(model_mem):
#                 pick_solution(i + 1, next_solution)
#             if i == len(model_mem) - 1:
#                 model_tp = []
#                 latency = 0
#                 flag = True
#                 for k, choice in enumerate(next_solution):
#                     # print(next_solution)
#                     rho, replica, stage = choice
#                     tp = (
#                         replica
#                         * stage
#                         * rho
#                         / (inf_latency[k] + (stage - 1) * com_latency[k])
#                     )
#                     latency += (inf_latency[k] + (stage - 1) * com_latency[k]) * rho
#                     flag &= (inf_latency[k] + (stage - 1) * com_latency[k]) < inf_latency[k] * 1.5
#                     model_tp.append(tp)
#                 if latency <= LATENCY_BOUND and flag:
#                     solution_space.append((min(model_tp), solution))
#                     print(len(solution_space))

def pick_solution(i, solution):
    if i >= len(model_mem):
        return
    for s in range(1, NUM_STAGE+1):
        next_solution = solution + [(ratio[i], s)]
        print(next_solution)
        if i < len(model_mem):
            pick_solution(i + 1, next_solution)
        if i == len(model_mem) - 1:
            model_tp = []
            latency = 0
            flag = True
            for k, choice in enumerate(next_solution):
                rho, stage = choice
                tp = (
                    stage
                    * rho
                    / (inf_latency[k] + (stage - 1) * com_latency[k])
                ) * 1000
                latency += (inf_latency[k] + (stage - 1) * com_latency[k]) * rho
                flag &= (inf_latency[k] + (stage - 1) * com_latency[k]) < inf_latency[k] * 1.5
                model_tp.append(tp)
            print(model_tp, np.array(model_tp) / np.array(ratio))
            lcm = np.lcm.reduce(np.ceil(np.array(model_tp) / np.array(ratio)).astype(int))
            replicas = [lcm / tp for tp in model_tp]
            mem = [model_mem[k] * r for k, r in enumerate(replicas)]
            replicas = np.ceil(np.array(replicas) / (np.sum(mem) / TOTAL_MEM)).astype(int).tolist()
            print(lcm, replicas, mem)

            if latency <= LATENCY_BOUND and flag:
                solution = [s + (replicas[k], ) for k, s in enumerate(solution)]
                solution_space.append((min(model_tp), solution))
                print(len(solution_space))


pick_solution(0, [])
print("solution_space", len(solution_space))




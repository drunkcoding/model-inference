from typing import Iterable
from tqdm import tqdm
import sys
import os
import threading
import requests
import numpy as np
import pandas as pd
import re
import subprocess
import io
from collections import deque
import multiprocessing as mp
import mysql.connector
import time
from operator import itemgetter

from responses import activate

from flask import Flask, request, jsonify, g

app = Flask(__name__)
DB_NAME = "serve.db"

connector_options = {"user": "admin", "host": "127.0.0.1", "database": "hyserve"}


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = mysql.connector.connect(**connector_options)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


# json={
#     "request_id": self.request_id,
#     "correlation_id": self.correlation_id,
#     "epos": self.config.epos,
#     "ppos": elf.config.ppos,
#     "type": self.config.type,
#     "start": request_start_time,
#     "end": request_end_time,
# },
QUERY_INTERVAL = 0.15
MICRO_INTERVAL = 1e-3
dtype = [("start", int), ("end", int)]

command = "nvidia-smi --query-gpu=index,uuid,gpu_bus_id --format=csv"
result = subprocess.run(command.split(), stdout=subprocess.PIPE)
df = pd.read_csv(
    io.StringIO(result.stdout.decode("utf-8")),
    index_col="index",
    converters={"uuid": str.strip},
)
df = df.sort_index()
df.iloc[:, 0] = df.iloc[:, 0].str.strip()
print(df)
print(df.columns)
# print(df.iloc[0, 0] == "GPU-0a0629f3-1b2a-f676-2b26-d91b9f0f5c61")
gpu_uuids = df.iloc[:, 0].to_list()

# if __name__ == "__main__":
con = mysql.connector.connect(**connector_options)
cur = con.cursor()
cur.execute(
    """CREATE TABLE IF NOT EXISTS measurement
        ( id VARCHAR(128) NOT NULL, 
        uuid VARCHAR(128), 
        start DOUBLE, 
        end DOUBLE, 
        energy DOUBLE, 
        power DOUBLE, 
        processed BOOLEAN, 
        UNIQUE(id) )"""
)
cur.execute(
    """CREATE TABLE IF NOT EXISTS request
            (id VARCHAR(128) NOT NULL, 
            uuid VARCHAR(128), 
            request_id INT UNSIGNED, 
            correlation_id INT UNSIGNED, 
            epos INT UNSIGNED, 
            ppos INT UNSIGNED, 
            type VARCHAR(128), 
            start DOUBLE, 
            end DOUBLE, 
            util DOUBLE,
            energy DOUBLE, 
            processed BOOLEAN,
            UNIQUE(id))"""
)
# cur.execute("DELETE FROM request WHERE type = 'none'")
cur.executemany(
    """INSERT INTO request VALUES (UUID(), %s, 0, 0, 0, 0, 'none', 0, 0, 0, 0, 0)""",
    [(uuid,) for uuid in gpu_uuids],
)
cur.executemany(
    """INSERT INTO request VALUES (UUID(), %s, 0, 0, 0, 0, 'void', 0, 0, 0, 0, 0)""",
    [(uuid,) for uuid in gpu_uuids],
)
con.commit()
con.close()


def db_metrics():
    last_data = None

    con = mysql.connector.connect(**connector_options)
    cur = con.cursor()

    print("db_metrics", os.getpid(), flush=True)
    last_timestamp = None
    while True:
        start_time = time.perf_counter()
        response = requests.get("http://localhost:8002/metrics")
        text = response.text
        energy_groups = re.findall(
            r'nv_energy_consumption{gpu_uuid="(.*)"} (\d+.\d+)', text
        )
        energy_groups = dict(energy_groups)
        power_groups = re.findall(
            r'nv_gpu_power_usage{gpu_uuid="(.*)"} (\d+.\d+)', text
        )
        power_groups = dict(power_groups)

        # print(energy_groups)
        measurement_list = []
        # end_time = time.perf_counter()
        for uuid in energy_groups:
            cum_energy = float(energy_groups[uuid])
            cur_energy = (
                -1 if last_data is None else cum_energy - float(last_data[uuid])
            )
            measurement_list.append(
                (
                    uuid,
                    last_timestamp + 1e-6 if last_timestamp is not None else start_time,
                    start_time,  # to distinguish start and last end
                    cur_energy,
                    float(power_groups[uuid]),
                )
            )
        if last_data is not None:
            for uuid in energy_groups:
                if last_data[uuid] == energy_groups[uuid]:
                    print("same reading", uuid, last_data[uuid], energy_groups[uuid])
                assert last_data[uuid] != energy_groups[uuid]
            cur.executemany(
                "INSERT INTO measurement VALUES (UUID(), %s, %s, %s, %s, %s, 0)",
                measurement_list,
            )
        last_data = energy_groups
        last_timestamp = start_time

        con.commit()
        end_time = time.perf_counter()
        # print(start_time, end_time - start_time)
        time.sleep(max(0, QUERY_INTERVAL - (end_time - start_time)))

def update_request_energy(cur, id, energy):
    if isinstance(id, Iterable):
        args = list(zip(energy, id))
        cur.executemany(
            """
            UPDATE request
            SET energy = energy + %s
            WHERE id = %s
            """,
            args,
        )
        return

    cur.execute(
        """
        UPDATE request
        SET energy = energy + %s
        WHERE id = %s
        """,
        (id, energy),
    )

def update_gpu_energy(cur, uuid, type, energy):
    cur.execute(
        """
        UPDATE request
        SET energy = energy + %s
        WHERE uuid = %s and type = %s
        """,
        (energy, uuid, type),
    )

# def update_gpu_energy(cur, uuid, type, energy, **kwds):
#     if "request_id" in kwds:
#         cur.execute(
#             """
#             UPDATE request
#             SET energy = energy + %s
#             WHERE uuid = %s and type = %s and request_id = %s and correlation_id = %s and epos = %s and ppos = %s
#             """,
#             (
#                 energy,
#                 uuid,
#                 type,
#                 kwds["request_id"],
#                 kwds["correlation_id"],
#                 kwds["epos"],
#                 kwds["ppos"],
#             ),
#         )
#     else:
#         cur.execute(
#             """
#             UPDATE request
#             SET energy = energy + %s
#             WHERE uuid = %s and type = %s
#             """,
#             (energy, uuid, type),
#         )


def share_energy(request_list, energy, active_set, holder):
    util_sum = 0
    for k in active_set:
        util_sum += request_list[k]["util"]
        holder[k] += energy * request_list[k]["util"]
    if util_sum > 1:
        print("util_sum", util_sum, active_set)
    assert util_sum <= 1
    # idle energy consumption
    for k in active_set:
        holder[k] += energy * (1 - util_sum) / len(active_set)

    return holder


LABEL_REQ_START = 0
LABEL_REQ_END = 1
LABEL_ENG_START = 2
LABEL_ENG_END = 3


def request_metrics():
    con = mysql.connector.connect(**connector_options)
    cur = con.cursor()

    print("request_metrics", os.getpid(), flush=True)

    while True:
        start_time = time.perf_counter()
        print(start_time, flush=True)

        cur.execute(
            "SELECT * FROM request WHERE processed = 0 and type != 'none' and type != 'void' ORDER BY start ASC"
        )
        req_rows = list(cur)
        req_names = [description[0] for description in cur.description]
        cur.execute("SELECT * FROM measurement WHERE processed = 0 ORDER BY start ASC")
        measure_rows = list(cur)
        measure_names = [description[0] for description in cur.description]

        # print([row[0] for row in measure_rows])
        print(len(req_rows), len(measure_rows), flush=True)

        gpu_energy = {
            uuid: [
                dict(zip(measure_names, list(row)))
                for row in measure_rows
                if row[1] == uuid
            ]
            for uuid in gpu_uuids
        }

        request_queue = {
            uuid: [
                dict(zip(req_names, list(row))) for row in req_rows if row[1] == uuid
            ]
            for uuid in gpu_uuids
        }

        # print("gpu_energy", gpu_energy)
        # print("request_queue", request_queue)

        for uuid in gpu_uuids:

            # TEST_GPU_SUM += gpu_energy[uuid][-1]["energy"]

            # if len(request_queue[uuid]) == 0:
            #     energy = gpu_energy[uuid][-1]["energy"]
            #     update_gpu_energy(cur, uuid, "void", energy)
            #     # TEST_MODEL_SUM += gpu_energy[uuid][-1]["energy"]
            #     # print("no request %s %s" % ("none", energy), flush=True)
            #     con.commit()
            #     continue

            # req_time = np.array(
            #     [(req["start"], req["end"]) for req in request_queue[uuid]], dtype=dtype
            # )
            # req_sort_idx = np.argsort(req_time, order=["start", "end"])

            request_list = request_queue[uuid]
            measure_list = gpu_energy[uuid]

            # print(measure_list)

            request_energy_holder = {idx: 0 for idx in range(len(request_list))}
            # request_energy_holder["none"] = 0

            all_timestamps = (
                [
                    (request["start"], LABEL_REQ_START, i)
                    for i, request in enumerate(request_list)
                ]
                + [
                    (request["end"], LABEL_REQ_END, i)
                    for i, request in enumerate(request_list)
                ]
                + [
                    (measure["start"], LABEL_ENG_START, i)
                    for i, measure in enumerate(measure_list)
                ]
                + [
                    (measure["end"], LABEL_ENG_END, i)
                    for i, measure in enumerate(measure_list)
                ]
            )
            all_timestamps = sorted(all_timestamps)
            # all_timestamps = np.sort(all_timestamps, axis=None)
            all_timestamps = list(zip(all_timestamps[:-1], all_timestamps[1:]))

            active_req_set = set()
            active_eng_set = set()
            finish_req_set = set()
            finsish_eng_set = set()
            for item_l, item_r in tqdm(all_timestamps, desc="all_timestamps"):
                time_l, label_l, idx_l = item_l
                time_r, label_r, idx_r = item_r

                print(item_l, item_r)

                if label_l == LABEL_REQ_START:
                    active_req_set.add(idx_l)
                # if label_r == LABEL_REQ_START:
                #     active_req_set.add(idx_r)
                if label_l == LABEL_ENG_START: 
                    active_eng_set.add(idx_l)
                # if label_r == LABEL_ENG_START:
                #     active_eng_set.add(idx_r)

                # print(active_req_set, active_eng_set)

                if label_l == LABEL_ENG_END and label_r == LABEL_ENG_START:
                    continue

                if len(active_eng_set) == 0:
                    print("no energy measured")
                    # active_req_set = set()
                    continue

                # # no request in this interval
                # if label_l == LABEL_ENG_START and label_r == LABEL_ENG_END:
                #     assert idx_l == idx_r
                #     energy = measure_list[idx_l]["energy"]
                #     update_gpu_energy(cur, uuid, "void", energy)

                idx = list(active_eng_set)[0]
                energy = (
                    measure_list[idx]["energy"]
                    * (time_r - time_l)
                    / (measure_list[idx]["end"] - measure_list[idx]["start"])
                )

                # no request in this micro-interval
                if (
                    (label_l == LABEL_ENG_START and label_r == LABEL_REQ_START)
                    or (label_l == LABEL_ENG_START and label_r == LABEL_ENG_END)
                    or (label_l == LABEL_REQ_END and label_r == LABEL_ENG_END)
                ):
                    if label_l == LABEL_ENG_START and label_r == LABEL_ENG_END:
                        update_gpu_energy(cur, uuid, "void", energy)
                    else:
                        update_gpu_energy(cur, uuid, "none", energy)

                # share request in this micro-interval
                else:
                    # if (label_l == LABEL_ENG_START and label_r == LABEL_REQ_START) or ():
                    # idx = list(active_eng_set)[0]
                    # energy = (
                    #     measure_list[idx]["energy"]
                    #     * (time_r - time_l)
                    #     / (measure_list[idx]["end"] - measure_list[idx]["start"])
                    # )
                    print(active_req_set, energy)
                    request_energy_holder = share_energy(
                        request_list, energy, active_req_set, request_energy_holder
                    )
                    print(request_energy_holder)
                    con.commit()

                # assert len(active_eng_set) == 1

                if label_r == LABEL_REQ_END:
                    active_req_set.remove(idx_r)
                    finish_req_set.add(idx_r)
                if label_r == LABEL_ENG_END:
                    active_eng_set.remove(idx_r)
                    finsish_eng_set.add(idx_r)

                # for idx in tqdm(request_energy_holder, desc="request_energy_holder"):
                #     energy = request_energy_holder[idx]
                #     request_list[idx]["energy"] = energy
            update_request_energy(cur, [request_list[i]['id'] for i in request_energy_holder.keys()], list(request_energy_holder.values()))
            con.commit()
            # completely finished
            cur.executemany(
                "UPDATE measurement SET processed = 1 WHERE id = %s",
                [(measure_list[i]["id"],) for i in finsish_eng_set],
            )
            cur.executemany(
                "UPDATE request SET processed = 1 WHERE id = %s",
                [(request_list[i]["id"],) for i in finish_req_set],
            )
            con.commit()

            # assert len(active_eng_set) == 0

            # partially finished modify timestamp

            # active_req_set = active_req_set.difference(finish_req_set)
            # active_eng_set = active_req_set.difference(finsish_eng_set)

            # idx = 0
            # energy_idx = 0
            # waiting_list = []
            # # for t_inv in tqdm(time_intervals, desc="time_intervals"):
            # #     time_start_bar = t_inv
            # #     time_end_bar = time_start_bar + MICRO_INTERVAL
            # for time_start_bar, time_end_bar in all_timestamps:
            #     micro_interval = time_end_bar - time_start_bar
            #     activate_req_set = []
            #     for i in range(idx, len(request_list)):
            #         if (
            #             request_list[i]["start"] >= time_start_bar
            #             and request_list[i]["end"] <= time_end_bar
            #         ):
            #             activate_req_set.append(i)
            #         if request_list[i]["start"] >= time_end_bar:
            #             break
            #     print(
            #         "activate_req_set",
            #         micro_interval,
            #         time_start_bar,
            #         time_end_bar,
            #         activate_req_set,
            #     )

            #     energy = 0
            #     energy_interval = 0
            #     for i in range(energy_idx, len(measure_list)):
            #         energy_idx = i
            #         query_interval = (
            #             measure_list[i]["end"] - measure_list[i]["start"]
            #         )
            #         query_energy = measure_list[i]["energy"]
            #         # interval completely enclosed in measurement
            #         if (
            #             measure_list[i]["start"] <= time_start_bar
            #             and measure_list[i]["end"] >= time_end_bar
            #         ):
            #             energy += micro_interval / query_interval * query_energy
            #             energy_interval += query_interval
            #             print(
            #                 "energy enclosed",
            #                 energy,
            #                 energy_interval,
            #                 query_energy,
            #                 measure_list[i]["end"],
            #                 measure_list[i]["start"],
            #             )
            #             break

            #         # interval cross boundary of measurement
            #         if i == len(measure_list) - 1:
            #             break
            #         if (
            #             measure_list[i]["end"] > time_start_bar
            #             and measure_list[i]["end"] < time_end_bar
            #         ):

            #             energy += (
            #                 (measure_list[i]["end"] - time_start_bar)
            #                 / query_interval
            #                 * query_energy
            #             )
            #             energy_interval += query_interval
            #             print(
            #                 "energy cross",
            #                 energy,
            #                 energy_interval,
            #                 measure_list[i]["end"],
            #                 measure_list[i]["start"],
            #             )
            #             # find latest query that contains the interval
            #             for k in range(i + 1, len(measure_list)):
            #                 query_interval = (
            #                     measure_list[k]["end"] - measure_list[k]["start"]
            #                 )
            #                 # if (
            #                 #     measure_list[k]["start"] > time_start_bar
            #                 #     and measure_list[k]["end"] < time_end_bar
            #                 # ):
            #                 #     energy += measure_list[k]["energy"]
            #                 #     energy_interval += query_interval
            #                 #     print("energy full", energy, energy_interval, measure_list[k]["end"], measure_list[k]["start"])
            #                 if (
            #                     measure_list[k]["end"] >= time_end_bar
            #                     and measure_list[k]["start"] < time_end_bar
            #                 ):
            #                     energy += (
            #                         (time_end_bar - measure_list[k]["start"])
            #                         / (
            #                             measure_list[k]["end"]
            #                             - measure_list[k]["start"]
            #                         )
            #                         * measure_list[k]["energy"]
            #                     )
            #                     energy_interval += query_interval
            #                     print(
            #                         "energy tail",
            #                         energy,
            #                         energy_interval,
            #                         measure_list[k]["end"],
            #                         measure_list[k]["start"],
            #                     )
            #                     break
            #             break
            #     print("energy", energy, energy_interval)

            #     # print(
            #     #     time_start_bar,
            #     #     time_start_bar,
            #     #     measure_list[0]["start"],
            #     #     measure_list[0]["end"],
            #     #     measure_list[-1]["start"],
            #     #     measure_list[-1]["end"],
            #     # )

            #     if energy == 0:
            #         # energy_idx = 0
            #         waiting_list += activate_req_set
            #         print("no energy measured")
            #         continue

            #     # energy_idx = energy_idx[0]
            #     if len(activate_req_set) > 0:
            #         idx = min(activate_req_set)

            #     if len(activate_req_set) == 0:
            #         request_energy_holder["none"] += energy
            #         continue

            #     # active energy consumption
            #     util_sum = 0
            #     for k in activate_req_set:
            #         util_sum += request_list[k]["util"]
            #         request_energy_holder[k] += energy * request_list[k]["util"]
            #     assert util_sum < 1
            #     # idle energy consumption
            #     for k in activate_req_set:
            #         request_energy_holder[k] += (
            #             energy * (1 - util_sum) / len(activate_req_set)
            #         )

            # # print("request_energy_holder", request_energy_holder, flush=True)

            # for idx in tqdm(request_energy_holder, desc="request_energy_holder"):
            #     if idx == "none":
            #         energy = request_energy_holder[idx]
            #         # print("has request %s %s" % ("none", energy), flush=True)
            #         update_gpu_energy(cur, uuid, "none", energy)
            #         # TEST_MODEL_SUM += energy
            #         continue
            #     energy = request_energy_holder[idx]
            #     # if energy == 0:
            #     #     print("error", idx, request_list[idx])
            #     # assert energy > 0
            #     request_list[idx]["energy"] = energy
            #     update_gpu_energy(cur, **request_list[idx])
            #     con.commit()
            #     # TEST_MODEL_SUM += energy

            # # gpu_energy[uuid] = gpu_energy[uuid][energy_idx:]
            # energy_idx = min(energy_idx, len(gpu_energy[uuid]) - 1)
            # cur.execute(
            #     "UPDATE measurement SET processed = 1 WHERE end <= %s", (time_end_bar,),
            # )
            # cur.execute(
            #     "UPDATE request SET processed = 1 WHERE end <= %s", (time_end_bar,),
            # )
            # if len(waiting_list) > 0:
            #     cur.executemany(
            #         "UPDATE request SET processed = 0 WHERE id = %s",
            #         [(request_list[k]["id"],) for k in waiting_list],
            #     )

            # con.commit()

        end_time = time.perf_counter()
        time.sleep(max(0, 1 - (end_time - start_time)))


@app.route("/meter/<uuid>", methods=["POST"])
def meter_handle(uuid):
    content = request.json
    con = get_db()
    cur = con.cursor()
    # print(content)
    cur.execute(
        """INSERT INTO request VALUES (UUID(), %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0)""",
        (uuid,) + tuple(list(content.values())),
    )
    con.commit()
    cur.close()
    # with lock:
    #     req_dict[uuid] += [content]
    # # print("meter_handle", req_dict)
    return jsonify({"uuid": uuid})


if __name__ == "__main__":
    mp.Process(target=db_metrics).start()
    time.sleep(0.5)
    mp.Process(target=request_metrics).start()
    app.run(host="0.0.0.0", port=10000, debug=False)


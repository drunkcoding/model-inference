from concurrent.futures.thread import ThreadPoolExecutor
from kazoo.client import KazooClient
import mysql.connector

import grpc
from mysql.connector import cursor
import numpy as np
import torch
from time import sleep
import torch.multiprocessing as mp
from ecosys.algo.calibratiuon import temperature_scale, temperature_scaling

from ecosys.algo.monte_carlo import monte_carlo_bounds
from ecosys.utils.message import deserialize
from .handler import RegisterModelHandler, ReportMetaHandler, ReportMetricsHandler
from protos.ecosys_pb2 import Message, RegisterModelRequest, RegisterModelResponse, RetCode

from protos.ecosys_pb2_grpc import CoordinatorServicer, add_CoordinatorServicer_to_server
from ecosys.utils.database import DatabaseHelper
from ecosys.context.arg_parser import ArgParser
from ecosys.context.srv_ctx import ServiceContext

args = ArgParser().parse()
ctx = ServiceContext(args.cfg)


class Coordinator(CoordinatorServicer):

    def __init__(self, ctx: ServiceContext):
        super(Coordinator, self).__init__()
        self.ctx = ctx
        self.executor = ThreadPoolExecutor(
            max_workers=self.ctx.cfg.srv_workers)

    def ReportMeta(self, request, context):
        handler = ReportMetaHandler(self.ctx)
        handler.req_msg.CopyFrom(request)
        handler.make_response()

        if not handler.check_req_type():
            return handler.rsp_msg

        # compute threshold

        conn = mysql.connector.connect(
            host=self.ctx.cfg.db_addr,
            user=self.ctx.cfg.db_user,
            database=self.ctx.cfg.db_name,
        )
        cursor = conn.cursor(dictionary=True)

        cursor.execute("UPDATE model_info SET labels=%s, outputs=%s WHERE model_name=%s",
                       (handler.req().labels, handler.req().outputs, handler.req().model_name))
        conn.commit()

        sleep(1)

        cursor.execute("SELECT model_name, energy, labels FROM model_info")
        waiting = False
        for row in cursor.fetchall():
            waiting |= row['labels'] == None
            waiting |= row['energy'] == None

        if waiting:
            handler.make_response_code(
                RetCode.ERR_DATA_INCOMPLETE, "need to wait for data arrival")
            self.ctx.logger.warn(
                "current updated %s, need to wait for more data arrival", handler.req().model_name)
        else:
            self.ctx.logger.info("start to compute confidence")
            mp.Process(target=self._compute_confidence).start()

        cursor.close()
        return handler.rsp_msg

    def _compute_confidence(self):
        sleep(5)
        conn = mysql.connector.connect(
            host=self.ctx.cfg.db_addr,
            user=self.ctx.cfg.db_user,
            database=self.ctx.cfg.db_name,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT model_name, energy, labels, outputs FROM model_info ORDER BY energy ASC")

        # waiting = False
        # for row in cursor.fetchall():
        #     waiting |= row['labels'] == None
        #     waiting |= row['energy'] == None
        #     waiting |= row['outputs'] == None

        # if waiting:
        #     self.ctx.logger.warn("need to wait for more data arrival")
        #     return
        
        model_probs = dict()
        model_energy = dict()
        n_models = 0
        num_labels = None

        model_keys = list()

        for row in cursor.fetchall():
            n_models += 1
            outputs = deserialize(
                row['outputs'], torch.device("cpu"))
            labels = deserialize(
                row['labels'], torch.device("cpu"))

            model_name = row['model_name']
            self.ctx.logger.info("%s temperature_scaling starting", model_name)
            temperature = temperature_scaling(outputs, labels)
            self.ctx.logger.info(
                "%s temperature_scaling finished %s ", model_name, temperature)
            cursor.execute("UPDATE model_info SET temperature=%s WHERE model_name=%s",
                           (temperature.item(), model_name))

            outputs = temperature_scale(outputs, temperature)
            m = torch.nn.Softmax(dim=1)

            prob, _ = torch.max(m(outputs), 1)
            pred_labels = torch.argmax(outputs, dim=1).flatten()
            model_probs[model_name] = np.array(
                [prob.tolist(), (labels == pred_labels).tolist()]
            ).T
            model_energy[model_name] = row['energy']
            num_labels = len(labels)
            model_keys.append(model_name)

        def total_reward(threshold):
            reward = 0
            energy = 0
            mask = np.array([False]*num_labels)
            for i, key in enumerate(model_keys):
                processed = (model_probs[key][~mask, 0] >= threshold[i]
                             ) if key in model_keys[:-1] else np.array([True]*num_labels)
                # correct_count = np.sum(
                #     model_probs[key][(~mask) & processed, 1])
                reward += np.around(np.sum(model_probs[key][(~mask) & processed, 1]) / 10.0) * 10
                # reward += np.around(correct_count /
                #                     (int(correct_count * 0.025) + 1)) * (int(correct_count * 0.025) + 1)
                energy += model_energy[key] * np.count_nonzero(~mask)
                mask |= processed
            return (reward, -energy)

        self.ctx.logger.info("threshold searching")
        threshold_bounds = monte_carlo_bounds(
            total_reward,
            [(0.5, 1.0)] * (n_models-1),
            [('reward', float), ('energy', float)],
            n=10000,
            tops=40,
            maxiter=15,
        )
        mc_threshold = np.min(
            threshold_bounds, axis=1
        )

        reward = 0
        energy = 0
        mask = np.array([False]*num_labels)
        for i, key in enumerate(model_keys):
            processed = (model_probs[key][~mask, 0] >= mc_threshold[i]
                            ) if key in model_keys[:-1] else np.array([True]*num_labels)
            reward += np.sum(model_probs[key][(~mask) & processed, 1])
            energy += model_energy[key] * np.count_nonzero(~mask)
            mask |= processed

        self.ctx.logger.info("threshold found %s, acc %s, energy %s", mc_threshold, reward / num_labels, energy)

        for i, key in enumerate(model_keys):
            cursor.execute("UPDATE model_info SET threshold=%s WHERE model_name=%s",
                           (mc_threshold[i] if key in model_keys[:-1] else 0.0, key))

        conn.commit()

    def ReportMetrics(self, request, context):
        handler = ReportMetricsHandler(self.ctx)
        handler.req_msg.CopyFrom(request)
        handler.make_response()

        if not handler.check_req_type():
            return handler.rsp_msg

        self.ctx.logger.debug(request)

        conn = mysql.connector.connect(
            host=self.ctx.cfg.db_addr,
            user=self.ctx.cfg.db_user,
            database=self.ctx.cfg.db_name,
        )
        cursor = conn.cursor(dictionary=True)

        # sql = f"INSERT INTO gpu_info VALUES ('%s', %f, %f, %f, %f, %f)"

        sql = f"INSERT INTO gpu_info \
            (\
                model_name, \
                power, \
                utilization, \
                mem_used, \
                mem_total, \
                num_query, \
                batch_size, \
                ctx_id\
            ) \
            VALUES (\
                '{handler.req().model_name}', \
                {handler.req().gpu_stats.power}, \
                {handler.req().gpu_stats.utilization}, \
                {handler.req().gpu_stats.mem_used}, \
                {handler.req().gpu_stats.mem_total}, \
                {handler.req().num_query}, \
                {handler.req().batch_size}, \
                {handler.req().ctx_id}\
            )"

        cursor.execute(sql)
        cursor.execute(
            f"CREATE OR REPLACE VIEW energy AS \
            SELECT power * (\
                SELECT (MAX(record_time) - MIN(record_time)) / SUM(num_query) \
                FROM gpu_info \
                WHERE model_name = '{handler.req().model_name}' and num_query > 0 and batch_size = {handler.req().batch_size} and ctx_id = {handler.req().ctx_id}) \
            FROM gpu_info WHERE model_name = '{handler.req().model_name}' and num_query > 0 and batch_size = {handler.req().batch_size} and ctx_id = {handler.req().ctx_id}"
        )
        cursor.execute(
            f"UPDATE model_info SET energy=(SELECT AVG(Name_exp_1) AS e FROM energy) WHERE model_name = '{handler.req().model_name}'")
        conn.commit()

        cursor.close()
        conn.close()
        return handler.rsp_msg

    def RegisterModel(self, request, context):
        handler = RegisterModelHandler(self.ctx)
        handler.req_msg.CopyFrom(request)
        handler.make_response()

        if not handler.check_req_type():
            return handler.rsp_msg

        conn = mysql.connector.connect(
            host=self.ctx.cfg.db_addr,
            user=self.ctx.cfg.db_user,
            database=self.ctx.cfg.db_name,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            f"SELECT energy, temperature, threshold FROM model_info WHERE model_name = '{handler.req().model_name}'")
        row = cursor.fetchone()

        self.ctx.logger.debug("%s row fetched from model_info %s", handler.req().model_name, row)

        # cursor.execute(
        #     f"DELETE FROM gpu_info WHERE model_name = '{handler.req().model_name}'")
        # self.db_helper.commit()

        if row is None:
            err_msg = "%s seeing the model first time, create entry only" % (
                handler.req().model_name)
            handler.make_response_code(RetCode.ERR_NOT_INITIALIZED, err_msg)
            self.ctx.logger.warn(err_msg)

            cursor.execute(
                f"INSERT INTO model_info (model_name) VALUES ('{handler.req().model_name}')")
            conn.commit()

            cursor.close()
            return handler.rsp_msg

        if row['energy'] is None or row['temperature'] is None or row['threshold'] is None:
            err_msg = "%s not measured or calibarated" % (
                handler.req().model_name)
            handler.make_response_code(RetCode.ERR_NOT_INITIALIZED, err_msg)
            self.ctx.logger.warn(err_msg)

            cursor.close()
            return handler.rsp_msg

        self.ctx.logger.info("%s already resigtered, threshold %s, temperature %s", handler.req(
        ).model_name, row['threshold'], row['temperature'])
        handler.rsp().threshold = row['threshold']
        handler.rsp().temperature = row['temperature']

        cursor.close()
        return handler.rsp_msg

    def serve(self):
        server = grpc.server(self.executor)
        add_CoordinatorServicer_to_server(self, server)
        server.add_insecure_port(self.ctx.cfg.srv_listen)
        server.start()
        server.wait_for_termination()

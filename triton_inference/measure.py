from multiprocessing.sharedctypes import Value
import subprocess
import re
import json
import multiprocessing as mp
import threading
import time
import os
from copy import deepcopy
import signal
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter


MEASURE_INTERVAL = 0.1

class ModelSelfReport(SummaryWriter):

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        super().__init__(log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue, flush_secs=flush_secs, filename_suffix=filename_suffix)

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        result = subprocess.run(["gpustat", "--json"], capture_output=True)
        gpustat = json.loads(result.stdout)
        gpus = gpustat['gpus']

        for gpu in gpus:
            if gpu["index"] == args['model_instance_device_id']: break

        self.gpu_uuid = gpu['uuid']

        tensorboard_path = os.path.join(args['model_repository'], "runs")
        self.__init__(tensorboard_path, filename_suffix=self.gpu_uuid)
        
        self.model_name = args['model_name']

        # self.model_instance_device_id = "cuda:" + args['model_instance_device_id']

        # self.model_repository = model_repository = args['model_repository']
        # model_version = args['model_version']
        # self.model_name = model_name = args['model_name']
        # model_name_or_path = os.path.join(
        #     os.path.join(model_repository, model_version),
        #     "model"
        # )

class ModelMetricsWriter():

    gpu_metric_names = [
        "nv_energy_consumption",
        "nv_gpu_power_limit",
        "nv_gpu_power_usage",
        "nv_gpu_memory_used_bytes",
        "nv_gpu_memory_total_bytes",
        "nv_gpu_utilization",
    ]

    model_metrics_names = [
        "nv_inference_request_success",
        "nv_inference_request_failure",
        "nv_inference_count",
        "nv_inference_exec_count",
        "nv_inference_request_duration_us",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_input_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_inference_compute_output_duration_us",
    ]

    cumulative_metrics = [
        "nv_inference_request_success",
        "nv_energy_consumption",
        "nv_inference_request_failure",
        "nv_inference_count",
        "nv_inference_exec_count",
        "nv_inference_request_duration_us",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_input_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_inference_compute_output_duration_us",
    ]


    request_based_metrics = [
        "nv_energy_consumption",
        "nv_inference_request_duration_us",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_input_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_inference_compute_output_duration_us",
    ]

    def __init__(self, folder, sub=None, text=None) -> None:
        if sub is not None: folder = os.path.join(folder, sub)
        self.writer = SummaryWriter(folder)
        self.text = text 
        self.step = 0

        self.base_metrics = None
        self.last_metrics = {}
        self.request_count = {}

        if text is not None:
            self.base_metrics = dict()
            for name in self.cumulative_metrics:
                if name in self.gpu_metric_names:
                    record = self.record_gpu_metric(name, accumulation=True, tensorboard=False)
                    self.base_metrics[name] = record
                if name in self.model_metrics_names:
                    record = self.record_model_metrics(name, accumulation=True, tensorboard=False)
                    self.base_metrics[name] = record
        # print(self.base_metrics)


    def write_metrics(self, name, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{name}/{key}", value, global_step=self.step)
    
    def update_metrics(self, metrics, key, value):
        if key in metrics:
            metrics[key] += float(value)
        else:
            metrics[key] = float(value)

    def reverse_accumulation(self, metrics, name):
        if self.base_metrics is not None and name in self.base_metrics:
            # print(self.base_metrics["nv_inference_request_success"], metrics)
            curr_metrics = deepcopy(metrics)
            for key in metrics: 
                metrics[key] -= self.base_metrics[name][key] 
                if "nv_inference_request_success" in self.last_metrics:
                    succ_count = self.last_metrics["nv_inference_request_success"]
                    if key in succ_count and succ_count[key] > 0:
                        metrics[key] /= succ_count[key] if name in self.request_based_metrics else self.step
            
            self.last_metrics[name] = deepcopy(metrics)
            self.base_metrics[name] = deepcopy(curr_metrics)

    def record_gpu_metric(self, name, accumulation=False, tensorboard=True):
        if name not in self.gpu_metric_names:
            raise ValueError()
        
        pattern = name + r'\{gpu_uuid="(.*)"\} (\d+.\d+)'
        matches = re.findall(pattern, self.text)

        # metrics = []
        metrics = {}
        for match in matches:
            self.update_metrics(metrics, match[0], float(match[-1]))
        if not accumulation:
            self.reverse_accumulation(metrics, name)
            # if match[0] in metrics:
            #     metrics[match[0]] += float(match[-1])
            # else:
            #     metrics[match[0]] = float(match[-1])
            # metrics.append({"gpu_uuid": match[0], name: float(match[-1])})
            # self.writer.add_scalar(f"{name}/{match[0]}", float(match[-1]))
        # self.writer.add_scalars(name, metrics)
        if tensorboard:
            self.write_metrics(name, metrics)
        return metrics

    def record_model_metrics(self, name, accumulation=False, tensorboard=True):
        if name not in self.model_metrics_names:
            raise ValueError()

        pattern = name + r'\{gpu_uuid="(.*)",model="(.*)",version="(\d+)"\} (\d+.\d+)'
        matches = re.findall(pattern, self.text)
        
        # per_gpu_metrics = {}
        # per_model_metrics = {}
        metrics = {}
        names = ["gpu_uuid", "model", "version", name] 
        for match in matches:
            self.update_metrics(metrics, match[0], float(match[-1]))
            self.update_metrics(metrics, match[1], float(match[-1]))
        if not accumulation:
            self.reverse_accumulation(metrics, name)
        if tensorboard:
            self.write_metrics(name, metrics)
        # self.write_metrics(name, per_model_metrics)
        return metrics

class ModelMetricsWriterBackend():

    remote = None

    def __init__(self, folder, sub=None) -> None:
        if sub is not None: folder = os.path.join(folder, sub)
        self.folder = folder
        self.step = 0

    def start(self):
        self.lock = threading.Lock()
        self.lock.acquire()
        # self.job = threading.Thread(target=self.monitor)
        self.job = mp.Process(target=self.monitor)
        self.job.start()
        self.pid = self.job.pid
        
    def monitor(self):
        # init value
        r = subprocess.run(['curl', f'http://{self.remote}:8002/metrics'], capture_output=True, text=True)
        writer = ModelMetricsWriter(self.folder, text=r.stdout)
        writer.step = self.step
        while self.lock.locked():
            r = subprocess.run(['curl', f'http://{self.remote}:8002/metrics'], capture_output=True, text=True)
            # self.f.write("timestamp %s\n" % time.time())
            # self.f.write(r.stdout)
            writer.text = r.stdout
            for metric in writer.gpu_metric_names:
                record = writer.record_gpu_metric(metric)
                # print(record)
            for metric in writer.model_metrics_names:
                record = writer.record_model_metrics(metric)
                # print(record)
            time.sleep(0.1)

        writer.writer.close()
    
    def stop(self):
        # self.event.clear()
        self.lock.release()
        # self.job.terminate()
        # self.job.join()

        os.kill(self.pid, signal.SIGKILL)

    # def nv_energy_consumption(self):
    #     pattern = r'nv_energy_consumption\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)

    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_energy_consumption": float(match[-1])})
    #         self.writer.add_scalar(f"nv_energy_consumption/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_gpu_power_limit(self):
    #     pattern = r'nv_gpu_power_limit\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_gpu_power_limit": float(match[-1])})
    #         self.writer.add_scalar(f"nv_gpu_power_limit/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_gpu_power_usage(self):
    #     pattern = r'nv_gpu_power_usage\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_gpu_power_usage": float(match[-1])})
    #         self.writer.add_scalar(f"nv_gpu_power_usage/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_gpu_memory_used_bytes(self):
    #     pattern = r'nv_gpu_memory_used_bytes\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_gpu_memory_used_bytes": float(match[-1])})
    #         self.writer.add_scalar(f"nv_gpu_memory_used_bytes/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_gpu_memory_total_bytes(self):
    #     pattern = r'nv_gpu_memory_total_bytes\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_gpu_memory_total_bytes": float(match[-1])})
    #         self.writer.add_scalar(f"nv_gpu_memory_total_bytes/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_gpu_utilization(self):
    #     pattern = r'nv_gpu_utilization\{gpu_uuid="(.*)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     for match in matches:
    #         metrics.append({"gpu_uuid": match[0], "nv_gpu_utilization": float(match[-1])})
    #         self.writer.add_scalar(f"nv_gpu_utilization/{match[0]}", float(match[-1]))
    #     return metrics

    # def nv_inference_request_duration_us(self):
    #     pattern = r'nv_inference_request_duration_us\{gpu_uuid="(.*)",model="(.*)",version="(\d+)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     names = ["gpu_uuid", "model", "version", "nv_inference_request_duration_us"] 
    #     for match in matches:
    #         match[-1] = float(match[-1])
    #         metric = dict(zip(names, match))
    #         metrics.append(metric)
    #         self.writer.add_scalars(f"nv_inference_request_duration_us/{match[0]}", {
    #             match[1]: float(match[-1])
    #         })
    #     return metrics

    # def nv_inference_count(self):
    #     pattern = r'nv_inference_count\{gpu_uuid="(.*)",model="(.*)",version="(\d+)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     names = ["gpu_uuid", "model", "version", "nv_inference_count"] 
    #     for match in matches:
    #         match[-1] = float(match[-1])
    #         metric = dict(zip(names, match))
    #         metrics.append(metric)
    #         self.writer.add_scalars(f"nv_inference_count/{match[0]}", {
    #             match[1]: float(match[-1])
    #         })
    #     return metrics

    # def nv_inference_count(self):
    #     pattern = r'nv_inference_count\{gpu_uuid="(.*)",model="(.*)",version="(\d+)"\} (\d+.\d+)'
    #     matches = re.findall(pattern, self.text)
        
    #     metrics = []
    #     names = ["gpu_uuid", "model", "version", "nv_inference_count"] 
    #     for match in matches:
    #         match[-1] = float(match[-1])
    #         metric = dict(zip(names, match))
    #         metrics.append(metric)
    #         self.writer.add_scalars(f"nv_inference_count/{match[0]}", {
    #             match[1]: float(match[-1])
    #         })
    #     return metrics

EXAMPLE = """
# HELP nv_inference_request_success Number of successful inference requests, all batch sizes
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="gpt_neo_cola_ensemble",version="1"} 1208.000000
nv_inference_request_success{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_request_success{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_request_success{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_request_success{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 1208.000000
nv_inference_request_success{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 1208.000000
nv_inference_request_success{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 1208.000000
nv_inference_request_success{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 1208.000000
nv_inference_request_success{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 151.000000
# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes
# TYPE nv_inference_request_failure counter
nv_inference_request_failure{model="gpt_neo_cola_ensemble",version="1"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 0.000000
nv_inference_request_failure{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 0.000000
# HELP nv_inference_count Number of inferences performed
# TYPE nv_inference_count counter
nv_inference_count{model="gpt_neo_cola_ensemble",version="1"} 1208.000000
nv_inference_count{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_count{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_count{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_count{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 1208.000000
nv_inference_count{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 1208.000000
nv_inference_count{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 1208.000000
nv_inference_count{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 1208.000000
nv_inference_count{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 151.000000
# HELP nv_inference_exec_count Number of model executions performed
# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="gpt_neo_cola_ensemble",version="1"} 1208.000000
nv_inference_exec_count{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_exec_count{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_exec_count{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 151.000000
nv_inference_exec_count{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 1208.000000
nv_inference_exec_count{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 1208.000000
nv_inference_exec_count{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 1208.000000
nv_inference_exec_count{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 1208.000000
nv_inference_exec_count{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 151.000000
# HELP nv_inference_request_duration_us Cummulative inference request duration in microseconds
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="gpt_neo_cola_ensemble",version="1"} 389818096.000000
nv_inference_request_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 47574974.000000
nv_inference_request_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 50371315.000000
nv_inference_request_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 47312693.000000
nv_inference_request_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 100372774.000000
nv_inference_request_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 101911347.000000
nv_inference_request_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 101666243.000000
nv_inference_request_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 85852877.000000
nv_inference_request_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 47603188.000000
# HELP nv_inference_queue_duration_us Cummulative inference queuing duration in microseconds
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="gpt_neo_cola_ensemble",version="1"} 89.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 2550.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 58356.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 2496.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 74012.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 76027.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 62096.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 18019.000000
nv_inference_queue_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 2554.000000
# HELP nv_inference_compute_input_duration_us Cummulative compute input duration in microseconds
# TYPE nv_inference_compute_input_duration_us counter
nv_inference_compute_input_duration_us{model="gpt_neo_cola_ensemble",version="1"} 4525558.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 1868.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 10202.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 1870.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 1510364.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 1503856.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 1491684.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 17834.000000
nv_inference_compute_input_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 1967.000000
# HELP nv_inference_compute_infer_duration_us Cummulative compute inference duration in microseconds
# TYPE nv_inference_compute_infer_duration_us counter
nv_inference_compute_infer_duration_us{model="gpt_neo_cola_ensemble",version="1"} 380808577.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 47563196.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 50292066.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 47301017.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 98721185.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 98894148.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 98580243.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 84611180.000000
nv_inference_compute_infer_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 47591319.000000
# HELP nv_inference_compute_output_duration_us Cummulative inference compute output duration in microseconds
# TYPE nv_inference_compute_output_duration_us counter
nv_inference_compute_output_duration_us{model="gpt_neo_cola_ensemble",version="1"} 4233759.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="distilgpt2_cola",version="0"} 7037.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="distilgpt2_cola",version="0"} 10354.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="distilgpt2_cola",version="0"} 6973.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a",model="gpt_neo_cola_part3",version="0"} 64549.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed",model="gpt_neo_cola_part2",version="0"} 1434554.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0",model="gpt_neo_cola_part1",version="0"} 1529611.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="gpt_neo_cola_part0",version="0"} 1203246.000000
nv_inference_compute_output_duration_us{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65",model="distilgpt2_cola",version="0"} 7025.000000
# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 0.000000
nv_gpu_utilization{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 0.000000
nv_gpu_utilization{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 0.000000
nv_gpu_utilization{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 0.000000
# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 16944988160.000000
nv_gpu_memory_total_bytes{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 16944988160.000000
nv_gpu_memory_total_bytes{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 16944988160.000000
nv_gpu_memory_total_bytes{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 16944988160.000000
# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 7451181056.000000
nv_gpu_memory_used_bytes{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 7111442432.000000
nv_gpu_memory_used_bytes{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 7111442432.000000
nv_gpu_memory_used_bytes{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 7329546240.000000
# HELP nv_gpu_power_usage GPU power usage in watts
# TYPE nv_gpu_power_usage gauge
nv_gpu_power_usage{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 58.954000
nv_gpu_power_usage{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 59.436000
nv_gpu_power_usage{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 60.320000
nv_gpu_power_usage{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 58.993000
# HELP nv_gpu_power_limit GPU power management limit in watts
# TYPE nv_gpu_power_limit gauge
nv_gpu_power_limit{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 300.000000
nv_gpu_power_limit{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 300.000000
nv_gpu_power_limit{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 300.000000
nv_gpu_power_limit{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 300.000000
# HELP nv_energy_consumption GPU energy consumption in joules since the Triton Server started
# TYPE nv_energy_consumption counter
nv_energy_consumption{gpu_uuid="GPU-71ce41fd-3ac0-afe7-c995-21119f1e984a"} 205620.582000
nv_energy_consumption{gpu_uuid="GPU-8674351b-fff7-56c3-f545-26313154b8ed"} 203679.455000
nv_energy_consumption{gpu_uuid="GPU-4bb61721-0ae2-9004-636f-e73bab9ef8e0"} 205479.234000
nv_energy_consumption{gpu_uuid="GPU-7a9c9f44-ff57-1177-e21a-1740a8cf1a65"} 199292.130000
"""

if __name__ == "__main__":

    writer = ModelMetricsWriter("tritonserver")
    writer.text = EXAMPLE
    print(writer.record_gpu_metric("nv_energy_consumption"))
    print(writer.record_gpu_metric("nv_gpu_power_limit"))
    print(writer.record_gpu_metric("nv_gpu_memory_total_bytes"))
    print(writer.record_gpu_metric("nv_gpu_memory_used_bytes"))
    print(writer.record_gpu_metric("nv_gpu_power_usage"))
    print(writer.record_gpu_metric("nv_gpu_utilization"))
    print(writer.record_gpu_metric("nv_energy_consumption"))

    print(writer.record_model_metrics("nv_inference_request_duration_us"))
    
    # print(writer.nv_energy_consumption())
    # print(writer.nv_gpu_power_limit())
    # print(writer.nv_gpu_memory_total_bytes())
    # print(writer.nv_gpu_memory_used_bytes())
    # print(writer.nv_gpu_power_usage())
    # print(writer.nv_gpu_utilization())
    # print(writer.nv_inference_request_duration_us())
    
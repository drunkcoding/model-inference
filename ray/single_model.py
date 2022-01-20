from ray import serve
from transformers import T5ForConditionalGeneration
import requests
import ray
import os


@serve.deployment(ray_actor_options={"num_gpus": 1})
class T5Model:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained(
            "google/t5-small-lm-adapt"
        ).to("cuda")

    def __call__(self, request):
        return {"result": 0}


serve.start()
T5Model.deploy()

resp = requests.post("http://127.0.0.1:8000/T5Model")
print(resp.json())

print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
# print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
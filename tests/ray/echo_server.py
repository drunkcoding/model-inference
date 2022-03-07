import ray
from ray import data, serve

ray.init(num_cpus=20, namespace=f"echo")
serve.start(detached=False)

@serve.deployment(max_concurrent_queries=100, route_prefix="/echo")
class EchoServer:
    def __init__(self):
        pass

    async def __call__(self, request):
        data = await request.json()
        return {
            "payload": [1.234] * 10
        }


EchoServer.options(
    num_replicas=4, name="echoserver", ray_actor_options={"num_cpus": 5}
).deploy()

while True:
    pass
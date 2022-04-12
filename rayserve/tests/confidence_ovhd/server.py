import time
import ray
from ray import serve
import logging
import numpy as np
from scipy.special import softmax

from hfutils.logger import Logger

@serve.deployment(max_concurrent_queries=100)
class HybridScheduler:
    def __init__(self):
        self.logger = Logger(__file__, logging.INFO, 50000000, 5)

    async def post_processing(self, ensemble_outputs, outputs, batch_mask, idx):
        size = ensemble_outputs.shape
        size = np.prod(size)

        ensemble_outputs = ensemble_outputs.copy()
        batch_mask = batch_mask.copy()
        
        start_time = time.perf_counter()
        local_mask = batch_mask[idx]
        ensemble_outputs = self.model_ensemble(
            ensemble_outputs, outputs, local_mask
        )

        extended_mask = self.offload_mask(
            ensemble_outputs, local_mask
        )

        # num_next_models = 1
        # if np.any(extended_mask) and num_next_models > 0:
        #     batch_mask[idx] &= ~extended_mask
        #     batch_mask[idx + 1] |= extended_mask
        #     # batch_mask = self.update_batch_mask(
        #     #     max_prob, batch_mask.copy(), extended_mask, idx
        #     # )
        #     # self.logger.trace(
        #     #     "%s batch_mask updated %s", options.name, batch_mask
        #     # )
        end_time = time.perf_counter()
        self.logger.info("size %s time %s (ms)", size, (end_time-start_time) * 1000)
        return ensemble_outputs, np.any(extended_mask)

    def offload_mask(self, logits, mask):
        probabilities = np.power(softmax(logits, axis=1), 2)
        prob_mask = np.all(probabilities < 0.5, axis=1)
        self.logger.debug(
            "(offload_mask) prob_mask %s %s", prob_mask, mask,
        )
        combined_mask = mask & prob_mask
        return combined_mask

    def model_ensemble(self, hist_outputs, outputs, mask):
        alpha = 0.6
        if hist_outputs is not None:
            hist_outputs[mask] = (
                hist_outputs[mask] * (1 - alpha)
                + outputs * alpha
            )
        else:
            hist_outputs = outputs.copy()
        return hist_outputs  # MEMCOPY MUTABLE

ray.init(address="ray://129.215.164.41:10001", namespace="confidence_ovhd")
serve.start(detached=True)


HybridScheduler.options(
    name="hybrid-scheduler",
    num_replicas=1,
    ray_actor_options={"num_cpus": 2},
).deploy()
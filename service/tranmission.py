import triton_python_backend_utils as pb_utils
import time
import torch
import json

from hfutils.logger import Logger
from hfutils.constants import np_to_torch_dtype


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

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

        self.logger = Logger(__file__, "info", 50000000, 5)
        self.is_start = "start" in args['model_name']
        self.device_id = int(args["model_instance_device_id"])
        self.device = torch.device("cuda:" + args["model_instance_device_id"])
        self.model_config = json.loads(args["model_config"])
        self.model_name = args["model_name"]

    async def execute(self, requests):
        responses = []

        exec_start_time = time.perf_counter()
        for request in requests:
            input = self.parse_input(request, "input")

            if self.is_start:
                intermediate = input.repeat((10,)).flatten()
            else:
                intermediate = input.reshape((10, -1))[0]
            
            output_tensors = [self.parse_output(intermediate, "output")]
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )
            responses.append(inference_response)

        exec_end_time = time.perf_counter()
        self.logger.info(
            "%s %s %s",
            exec_start_time, exec_end_time,
            exec_end_time - exec_start_time,
        )
        return responses

    def parse_input(self, request, field):
        # self.logger.debug("parse_input: request %s, field %s", request, field)
        input = pb_utils.get_input_tensor_by_name(request, field)
        if input is None:
            return
        # input = from_dlpack(input.to_dlpack())
        input = input.as_numpy()
        input = torch.as_tensor(
            input, dtype=np_to_torch_dtype(input.dtype), device=self.device
        )
        return input

    def parse_output(self, output, field):
        # return pb_utils.Tensor.from_dlpack(field, to_dlpack(output))
        output_config = pb_utils.get_output_config_by_name(self.model_config, field)
        output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        output_pb = pb_utils.Tensor(field, output.astype(output_dtype))
        return output_pb

    def finalize(self):
        # self.tracker.stop()
        print("Cleaning up...")

import torch
import time

ones = torch.ones((1000,1000,1000))

time.sleep(1)

ones = ones.to("cuda:1")
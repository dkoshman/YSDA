import torch

import my_ml_tools.utils as utils

def test_free_cuda():
    memory_before = torch.cuda.mem_get_info()
    print(memory_before, type(memory_before))
    utils.free_cuda()
    memory_after = torch.cuda.mem_get_info()
    assert memory_after >= memory_before

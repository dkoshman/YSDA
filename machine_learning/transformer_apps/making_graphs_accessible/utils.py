import os
import pickle
from typing import Callable, TypeVar

T = TypeVar("T")


def set_tokenizers_parallelism(enable: bool):
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if enable else "false"


def set_torch_device_order_pci_bus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def load_pickle_or_build_object_and_save(
    pickle_path: str, build_object: Callable[[], T], overwrite=False
) -> T:
    if overwrite or not os.path.exists(pickle_path):
        pickle.dump(build_object(), open(pickle_path, "wb"))
    else:
        print(f"Reusing object {pickle_path}.")
    return pickle.load(open(pickle_path, "rb"))

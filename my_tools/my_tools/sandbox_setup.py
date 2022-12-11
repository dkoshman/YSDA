# Expected use case of this module for notebooks:
# %load_ext nb_black
# from my_tools.sandbox_setup import *

import abc
import asyncio
import contextlib
import enum
import functools
import io
import os
import re
import requests
import sys
import time
import types
import typing

import catboost
import einops
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import sklearn
import torch
import wandb
import yaml

from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm.auto import tqdm


tensor = torch.arange(12).reshape(3, 4).to(torch.float32)
dataframe = pd.DataFrame(
    dict(
        a=np.array([7, 42]).repeat(2),
        b=["red", "black", "black", "red"],
        c=1 / np.arange(4, 0, -1),
    ),
    index=pd.MultiIndex.from_arrays(
        [np.arange(2).repeat(2), np.tile(np.arange(2), 2)], names=["first", "second"]
    ),
)

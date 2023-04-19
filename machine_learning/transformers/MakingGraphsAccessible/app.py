import functools

import gradio

from machine_learning.transformers.MakingGraphsAccessible.config import CONFIG
from machine_learning.transformers.MakingGraphsAccessible.model import (
    predict_string,
    build_model,
)


def main():
    config = CONFIG
    config.pretrained_model_name = "checkpoint"
    model = build_model(config)

    interface = gradio.Interface(
        title="Making graphs accessible",
        description="Generate textual representation of a graph\n"
        "https://www.kaggle.com/competitions/benetech-making-graphs-accessible",
        fn=functools.partial(predict_string, model=model),
        inputs="image",
        outputs="text",
        examples="examples",
    )

    interface.launch()

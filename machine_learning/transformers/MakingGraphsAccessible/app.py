import gradio
import pandas as pd
from matplotlib import pyplot as plt

from config import CONFIG
from machine_learning.transformers.MakingGraphsAccessible.data import (
    get_extra_tokens,
    BenetechOutput,
    ChartType,
)
from model import predict_string, build_model


def gradio_visualize_prediction(string):
    string = string.removeprefix(get_extra_tokens().benetech_prompt)

    if not BenetechOutput.does_string_match_expected_pattern(string):
        return

    benetech_output = BenetechOutput.from_string(string)
    x = benetech_output.x_data[: len(benetech_output.y_data)]
    y = benetech_output.y_data[: len(benetech_output.x_data)]

    df = pd.DataFrame(dict(x=x, y=y))

    plt_plot = {
        ChartType.line: plt.plot,
        ChartType.scatter: plt.scatter,
        ChartType.horizontal_bar: plt.barh,
        ChartType.vertical_bar: plt.bar,
        ChartType.dot: plt.scatter,
    }

    plt_plot[benetech_output.chart_type](x, y)
    plt.xticks(rotation=30)
    plt.savefig("plot.png")

    ...


def main():
    config = CONFIG
    config.pretrained_model_name = "checkpoint"
    model = build_model(config)

    interface = gradio.Interface(
        title="Making graphs accessible",
        description="Generate textual representation of a graph\n"
                    "https://www.kaggle.com/competitions/benetech-making-graphs-accessible",
        fn=lambda image: predict_string(image, model),
        inputs="image",
        outputs="text",
        examples="examples",
    )

    interface.launch()

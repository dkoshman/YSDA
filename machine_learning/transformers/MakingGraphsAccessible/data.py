import dataclasses
import enum
import functools
import json
import os
import re
import types
from typing import Callable

import einops
import imageio
import numpy as np
import torch.utils.data
import torchvision
import tqdm

from config import CONFIG
from utils import load_pickle_or_build_object_and_save


class Source(enum.Enum):
    generated = "generated"
    extracted = "extracted"


class ChartType(enum.Enum):
    dot = "dot"
    horizontal_bar = "horizontal_bar"
    vertical_bar = "vertical_bar"
    line = "line"
    scatter = "scatter"


@dataclasses.dataclass
class PlotBoundingBox:
    height: int
    width: int
    x0: int
    y0: int

    def get_bounds(self):
        xs = [self.x0, self.x0 + self.width, self.x0 + self.width, self.x0, self.x0]
        ys = [self.y0, self.y0, self.y0 + self.height, self.y0 + self.height, self.y0]
        return xs, ys


@dataclasses.dataclass
class DataPoint:
    x: float or str
    y: float or str


class TextRole(enum.Enum):
    axis_title = "axis_title"
    chart_title = "chart_title"
    legend_label = "legend_label"
    tick_grouping = "tick_grouping"
    tick_label = "tick_label"
    other = "other"


@dataclasses.dataclass
class Polygon:
    x0: int
    x1: int
    x2: int
    x3: int
    y0: int
    y1: int
    y2: int
    y3: int

    def get_bounds(self):
        xs = [
            self.x0,
            self.x1,
            self.x2,
            self.x3,
            self.x0,
        ]
        ys = [
            self.y0,
            self.y1,
            self.y2,
            self.y3,
            self.y0,
        ]
        return xs, ys


@dataclasses.dataclass
class Text:
    id: int
    polygon: Polygon
    role: TextRole
    text: str

    def __post_init__(self):
        self.polygon = Polygon(**self.polygon)
        self.role = TextRole(self.role)


class ValuesType(enum.Enum):
    categorical = "categorical"
    numerical = "numerical"


@dataclasses.dataclass
class Tick:
    id: int
    x: int
    y: int


class TickType(enum.Enum):
    markers = "markers"
    separators = "separators"


@dataclasses.dataclass
class Axis:
    values_type: ValuesType
    tick_type: TickType
    ticks: list[Tick]

    def __post_init__(self):
        self.values_type = ValuesType(self.values_type)
        self.tick_type = TickType(self.tick_type)
        self.ticks = [
            Tick(id=kw["id"], x=kw["tick_pt"]["x"], y=kw["tick_pt"]["y"])
            for kw in self.ticks
        ]

    def get_bounds(self):
        min_x = min(tick.x for tick in self.ticks)
        max_x = max(tick.x for tick in self.ticks)
        min_y = min(tick.y for tick in self.ticks)
        max_y = max(tick.y for tick in self.ticks)
        xs = [min_x, max_x, max_x, min_x, min_x]
        ys = [min_y, min_y, max_y, max_y, min_y]
        return xs, ys


def convert_dashes_to_underscores_in_key_names(dictionary):
    return {k.replace("-", "_"): v for k, v in dictionary.items()}


@dataclasses.dataclass
class Axes:
    x_axis: Axis
    y_axis: Axis

    def __post_init__(self):
        self.x_axis = Axis(**convert_dashes_to_underscores_in_key_names(self.x_axis))
        self.y_axis = Axis(**convert_dashes_to_underscores_in_key_names(self.y_axis))


def preprocess_numerical_value(value):
    value = float(value)
    value = 0 if np.isnan(value) else value
    return value


def preprocess_value(value, value_type: ValuesType):
    if value_type == ValuesType.numerical:
        return preprocess_numerical_value(value)
    else:
        return str(value)


@dataclasses.dataclass
class Annotation:
    source: Source
    chart_type: ChartType
    plot_bb: PlotBoundingBox
    text: list[Text]
    axes: Axes
    data_series: list[DataPoint]

    def __post_init__(self):
        self.source = Source(self.source)
        self.chart_type = ChartType(self.chart_type)
        self.plot_bb = PlotBoundingBox(**self.plot_bb)
        self.text = [Text(**kw) for kw in self.text]
        self.axes = Axes(**convert_dashes_to_underscores_in_key_names(self.axes))
        self.data_series = [DataPoint(**kw) for kw in self.data_series]

        for i in range(len(self.data_series)):
            self.data_series[i].x = preprocess_value(
                self.data_series[i].x, self.axes.x_axis.values_type
            )
            self.data_series[i].y = preprocess_value(
                self.data_series[i].y, self.axes.y_axis.values_type
            )

    @staticmethod
    def from_dict_with_dashes(kwargs):
        return Annotation(**convert_dashes_to_underscores_in_key_names(kwargs))

    @staticmethod
    def from_image_index(image_index: int):
        image_id = load_train_image_ids()[image_index]
        return Annotation.from_dict_with_dashes(load_image_annotation(image_id))

    def get_text_by_role(self, text_role: TextRole) -> list[Text]:
        return [t for t in self.text if t.role == text_role]


@dataclasses.dataclass
class AnnotatedImage:
    id: str
    image: np.ndarray
    annotation: Annotation

    @staticmethod
    def from_image_id(image_id: str):
        return AnnotatedImage(
            id=image_id,
            image=load_image(image_id),
            annotation=Annotation.from_dict_with_dashes(
                load_image_annotation(image_id)
            ),
        )

    @staticmethod
    def from_image_index(image_index: int):
        return AnnotatedImage.from_image_id(load_train_image_ids()[image_index])


def generate_annotated_images():
    for image_id in tqdm.autonotebook.tqdm(
        load_train_image_ids(), "Iterating over annotated images"
    ):
        yield AnnotatedImage.from_image_id(image_id)


@functools.cache
def load_train_image_ids() -> list[str]:
    train_image_ids = [i.replace(".jpg", "") for i in os.listdir("data/train/images")]
    return train_image_ids[: 1000 if CONFIG.debug else None]


@functools.cache
def load_test_image_ids() -> list[str]:
    return [i.replace(".jpg", "") for i in os.listdir("data/test/images")]


@functools.cache
def load_image_annotation(image_id: str) -> dict:
    return json.load(open(f"data/train/annotations/{image_id}.json"))


def load_image(image_id: str) -> np.ndarray:
    return imageio.v3.imread(open(f"data/train/images/{image_id}.jpg", "rb"))


@dataclasses.dataclass
class DataItem:
    image: torch.FloatTensor
    target_string: str
    data_index: int

    def __post_init__(self):
        shape = einops.parse_shape(self.image, "channel height width")
        assert shape["channel"] == 3, "Image is expected to have 3 channels."


def split_train_indices_by_source():
    extracted_image_indices = []
    generated_image_indices = []
    for i, annotated_image in enumerate(generate_annotated_images()):
        if annotated_image.annotation.source == Source.extracted:
            extracted_image_indices.append(i)
        else:
            generated_image_indices.append(i)
    return extracted_image_indices, generated_image_indices


def get_train_val_split_indices(val_fraction=0.1, seed=42):
    np.random.seed(seed)
    val_size = int(len(load_train_image_ids()) * val_fraction)

    extracted_image_indices, generated_image_indices = split_train_indices_by_source()
    extracted_image_indices = np.random.permutation(extracted_image_indices)
    generated_image_indices = np.random.permutation(generated_image_indices)

    val_indices = extracted_image_indices[:val_size]
    n_generated_images_in_val = val_size - len(val_indices)
    val_indices = np.concatenate(
        [val_indices, generated_image_indices[:n_generated_images_in_val]]
    )

    train_indices = generated_image_indices[n_generated_images_in_val:]

    assert len(set(train_indices) | set(val_indices)) == len(load_train_image_ids())
    assert len(val_indices) == val_size
    assert len(set(train_indices) & set(val_indices)) == 0

    return train_indices, val_indices


def to_token_str(value: str or enum.Enum):
    string = value.name if isinstance(value, enum.Enum) else value
    if re.fullmatch("<.*>", string):
        return string
    else:
        return f"<{string}>"


@functools.cache
def get_extra_tokens() -> types.SimpleNamespace:
    token_ns = types.SimpleNamespace()

    token_ns.benetech_prompt = to_token_str("benetech_prompt")
    token_ns.benetech_prompt_end = to_token_str("/benetech_prompt")
    token_ns.x_start = to_token_str("x_start")
    token_ns.y_start = to_token_str("y_start")
    token_ns.value_separator = to_token_str(";")

    for chart_type in ChartType:
        setattr(token_ns, chart_type.name, to_token_str(chart_type))

    for values_type in ValuesType:
        setattr(token_ns, values_type.name, to_token_str(values_type))

    return token_ns


def convert_number_to_scientific_string(value: int or float) -> str:
    return f"{value:.{CONFIG.float_scientific_notation_string_precision}e}"


def convert_axis_data_to_string(
    axis_data: list[str or float], values_type: ValuesType
) -> str:
    formatted_axis_data = []
    for value in axis_data:
        if values_type == ValuesType.numerical:
            value = convert_number_to_scientific_string(value)
        formatted_axis_data.append(value)
    return get_extra_tokens().value_separator.join(formatted_axis_data)


def convert_string_to_axis_data(string, values_type: ValuesType):
    data = string.split(get_extra_tokens().value_separator)
    if values_type == ValuesType.numerical:
        data = [float(i.replace(" ", "")) for i in data]
    return data


@dataclasses.dataclass
class BenetechOutput:
    chart_type: ChartType
    x_values_type: ValuesType
    y_values_type: ValuesType
    x_data: list[str or float]
    y_data: list[str or float]

    def __post_init__(self):
        self.chart_type = ChartType(self.chart_type)
        self.x_values_type = ValuesType(self.x_values_type)
        self.y_values_type = ValuesType(self.y_values_type)
        assert isinstance(self.x_data, list)
        assert isinstance(self.y_data, list)

    def get_main_characteristics(self):
        return (
            self.chart_type,
            self.x_values_type,
            self.y_values_type,
            len(self.x_data),
            len(self.y_data),
        )

    @staticmethod
    def from_annotation(annotation: Annotation):
        return BenetechOutput(
            chart_type=annotation.chart_type,
            x_values_type=annotation.axes.x_axis.values_type,
            y_values_type=annotation.axes.y_axis.values_type,
            x_data=[dp.x for dp in annotation.data_series],
            y_data=[dp.y for dp in annotation.data_series],
        )

    def to_string(self):
        return self.format_strings(
            chart_type=self.chart_type,
            x_values_type=self.x_values_type,
            y_values_type=self.y_values_type,
            x_data=convert_axis_data_to_string(self.x_data, self.x_values_type),
            y_data=convert_axis_data_to_string(self.y_data, self.y_values_type),
        )

    @staticmethod
    def format_strings(*, chart_type, x_values_type, y_values_type, x_data, y_data):
        chart_type = to_token_str(chart_type)
        x_values_type = to_token_str(x_values_type)
        y_values_type = to_token_str(y_values_type)
        token_ns = get_extra_tokens()
        return (
            f"{token_ns.benetech_prompt}{chart_type}"
            f"{token_ns.x_start}{x_values_type}{x_data}"
            f"{token_ns.y_start}{y_values_type}{y_data}"
            f"{token_ns.benetech_prompt_end}"
        )

    @staticmethod
    def get_string_pattern():
        field_names = [field.name for field in dataclasses.fields(BenetechOutput)]
        pattern = BenetechOutput.format_strings(
            **{field_name: f"(?P<{field_name}>.*?)" for field_name in field_names}
        )
        return pattern

    @staticmethod
    def does_string_match_expected_pattern(string):
        try:
            BenetechOutput.from_string(string)
            return True
        except:
            return False

    @staticmethod
    def from_string(string):
        fullmatch = re.fullmatch(BenetechOutput.get_string_pattern(), string)
        benetech_kwargs = fullmatch.groupdict()
        benetech_kwargs["chart_type"] = ChartType(benetech_kwargs["chart_type"])
        benetech_kwargs["x_values_type"] = ValuesType(benetech_kwargs["x_values_type"])
        benetech_kwargs["y_values_type"] = ValuesType(benetech_kwargs["y_values_type"])
        benetech_kwargs["x_data"] = convert_string_to_axis_data(
            benetech_kwargs["x_data"], benetech_kwargs["x_values_type"]
        )
        benetech_kwargs["y_data"] = convert_string_to_axis_data(
            benetech_kwargs["y_data"], benetech_kwargs["y_values_type"]
        )
        return BenetechOutput(**benetech_kwargs)


def get_annotation_ground_truth_str(annotation: Annotation):
    benetech_output = BenetechOutput(
        chart_type=annotation.chart_type,
        x_values_type=annotation.axes.x_axis.values_type,
        x_data=[dp.x for dp in annotation.data_series],
        y_values_type=annotation.axes.y_axis.values_type,
        y_data=[dp.y for dp in annotation.data_series],
    )
    return benetech_output.to_string()


def get_annotation_ground_truth_str_from_image_index(image_index: int) -> str:
    return get_annotation_ground_truth_str(Annotation.from_image_index(image_index))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, indices: list[int]):
        super().__init__()
        self.indices = indices
        self.to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> DataItem:
        data_index = self.indices[idx]

        annotated_image = AnnotatedImage.from_image_index(data_index)

        image = annotated_image.image
        image = self.to_tensor(image)

        target_string = get_annotation_ground_truth_str(annotated_image.annotation)

        return DataItem(image=image, target_string=target_string, data_index=data_index)


def get_train_val_datasets():
    train_indices, val_indices = load_pickle_or_build_object_and_save(
        CONFIG.train_val_indices_path,
        lambda: get_train_val_split_indices(CONFIG.val_fraction, CONFIG.seed),
    )
    return Dataset(train_indices), Dataset(val_indices)


def get_train_dataset():
    return get_train_val_datasets()[0]


def get_val_dataset():
    return get_train_val_datasets()[1]


@dataclasses.dataclass
class Batch:
    images: torch.FloatTensor
    labels: torch.IntTensor
    data_indices: list[int]

    def __post_init__(self):
        if CONFIG.debug:
            images_shape = einops.parse_shape(self.images, "batch channel height width")
            labels_shape = einops.parse_shape(self.labels, "batch label")
            assert images_shape["batch"] == labels_shape["batch"]
            assert len(self.data_indices) == images_shape["batch"]


class Split(enum.Enum):
    train = "train"
    val = "val"


BatchCollateFunction = Callable[[list[DataItem], Split], Batch]


def build_dataloader(split: Split, batch_collate_function: BatchCollateFunction):
    return torch.utils.data.DataLoader(
        get_train_dataset() if split == Split.train else get_val_dataset(),
        batch_size=CONFIG.batch_size,
        shuffle=split == Split.train,
        num_workers=CONFIG.num_workers,
        collate_fn=functools.partial(batch_collate_function, split=split),
    )

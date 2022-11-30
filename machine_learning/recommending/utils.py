import abc
import collections
import contextlib
import functools
import inspect
import io
import os
import sys
import time
import warnings
from types import TracebackType, FunctionType
from typing import TextIO, Callable, Type

import line_profiler
import shap
import torch
import wandb
import yaml
from matplotlib import pyplot as plt

from torch.utils.data import random_split


def build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


def split_dataset(dataset, fraction):
    len_dataset = len(dataset)
    right_size = int(fraction * len_dataset)
    left_size = len_dataset - right_size
    return random_split(dataset, [left_size, right_size])


def fetch_artifact(
    *, entity=None, project=None, artifact_name, alias="latest", api_key=None
):
    wandb_api = wandb.Api(api_key=api_key)
    entity = entity or wandb.run.entity
    project = project or wandb.run.project
    artifact = wandb_api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
    return artifact


def load_path_from_artifact(artifact, path_inside_artifact="checkpoint"):
    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(
        artifact_dir, artifact.get_path(path_inside_artifact).path
    )
    return checkpoint_path


def update_from_base_config(config, base_config_file):
    """Keeps everything from config, and updates it with entries from base config up to 1 depth in."""
    base_config = yaml.safe_load(open(base_config_file))
    for k, v in config.items():
        if k in base_config and isinstance(v, dict):
            base_config[k].update(v)
        else:
            base_config[k] = v
    return base_config


def wandb_context_manager(config):
    if wandb.run is None and config.get("logger") is not None:
        return wandb.init(project=config.get("project"), config=config)
    return contextlib.nullcontext()


@contextlib.contextmanager
def plt_figure(*args, title=None, **kwargs):
    figure = plt.figure(*args, **kwargs)
    if title is not None:
        plt.title(title)
    try:
        yield figure
    finally:
        plt.close(figure)


@contextlib.contextmanager
def wandb_plt_figure(title, log: bool = True, *args, **kwargs):
    with plt_figure(*args, title=title, **kwargs) as figure:
        yield figure
        if log:
            wandb.log({title: wandb.Image(figure)})


def tensor_size_in_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def batch_size_in_bytes(batch):
    if torch.is_tensor(batch):
        return tensor_size_in_bytes(batch)
    elif isinstance(batch, dict):
        return sum(batch_size_in_bytes(i) for i in batch.values())
    return 0


@contextlib.contextmanager
def filter_warnings(action: str, category=None):
    with warnings.catch_warnings():
        warnings.simplefilter(action=action, category=category)
        yield


def save_shap_force_plot(shap_plot: shap.plots._force.BaseVisualizer) -> TextIO:
    textio = io.TextIOWrapper(io.BytesIO())
    shap.save_html(textio, shap_plot)
    textio.seek(0)
    return textio


class Singleton(type):
    """
    class A(metaclass=Singleton):
        pass

    a1 = A()
    a2 = A()
    a1 is a2
    True
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    profilers_dict = collections.defaultdict(
        lambda: dict(profiler=line_profiler.LineProfiler(), accumulated_logs=0)
    )

    def log(self, dict_to_log: dict) -> None:
        raise NotImplementedError

    def log_timer(self, timer: "Timer") -> None:
        raise NotImplementedError

    def register_function(self, function: FunctionType) -> None:
        if function not in self.profilers_dict:
            profiler = self.profilers_dict[function]["profiler"]
            profiler.add_function(function)
            profiler.enable()

    def register_function_call(self, function, log_every=1) -> None:
        profiler_dict = self.profilers_dict[function]
        profiler_dict["accumulated_logs"] += 1
        if profiler_dict["accumulated_logs"] >= log_every:
            profiler_dict["accumulated_logs"] = 0
            self.log_profiled_function(
                profiler=profiler_dict["profiler"], function=function
            )

    def log_profiled_function(
        self, profiler: line_profiler.LineProfiler, function: FunctionType
    ) -> None:
        raise NotImplementedError


class FileLogger(Logger):
    def __init__(self, file: io.TextIOBase = sys.stderr):
        self.file = file

    # TODO:color
    def log(self, dict_to_log: dict):
        print(*[f"{k}:\t{v}" for k, v in dict_to_log.items()], sep="\n", file=self.file)

    def log_timer(self, timer):
        print(f"{timer.get_name()}:\t{timer.delta}", file=self.file)

    def log_profiled_function(
        self, profiler: line_profiler.LineProfiler, function: FunctionType
    ):
        profiler.print_stats(stream=self.file)


class WandbLogger(Logger):
    total_times = collections.defaultdict(lambda: 0)

    def log(self, dict_to_log: dict):
        if wandb.run is not None:
            wandb.log(dict_to_log)

    def log_timer(self, timer: "Timer"):
        name = timer.get_name()
        time_name = f"time/{name}"
        total_time_name = f"time/total/{name}"
        if wandb.run is not None:
            wandb.define_metric(time_name, summary="mean")
            wandb.define_metric(total_time_name, summary="last")
        self.total_times[name] += timer.delta
        self.log({time_name: timer.delta, total_time_name: self.total_times[name]})

    def log_profiled_function(
        self, profiler: line_profiler.LineProfiler, function: FunctionType
    ):
        stream = io.StringIO()
        profiler.print_stats(stream=stream)
        stream.seek(0)
        title = f"profile/{function.__qualname__}"
        wandb.log({title: wandb.Html(f"<pre>{stream.read()}</pre>")})


class DecoratorInterface:
    function: Callable = None

    def on_enter(self) -> None:
        """On enter hook."""

    def on_exit(self) -> None:
        """On exit hook."""

    def on_before_function_call(self, *args, **kwargs) -> None:
        """A hook, args and kwargs are the ones that were passed to the function."""

    def on_after_function_call(self, result, *args, **kwargs) -> None:
        """A hook, function returned result when it was called with args and kwargs."""

    def decorate(self, function: Callable) -> Callable:
        @functools.wraps(function)
        def wrap(*args, **kwargs):
            self.function = function
            self.on_enter()
            self.on_before_function_call(*args, **kwargs)
            result = function(*args, **kwargs)
            self.on_after_function_call(result, *args, **kwargs)
            self.on_exit()
            self.function = None
            return result

        return wrap

    __call__ = decorate


class Timer(DecoratorInterface):
    """Context manager and decorator for timing."""

    def __init__(self, name: str = "", logger: Logger = WandbLogger()):
        self.name = name
        self.logger = logger
        self.begin: float or None = None
        self.end: float or None = None

    def get_name(self):
        return self.name or (
            self.function.__qualname__ if self.function is not None else ""
        )

    @property
    def delta(self) -> float or None:
        return None if self.end is None else self.end - self.begin

    def on_enter(self):
        self.begin = time.time()
        self.end = None

    def on_exit(self):
        self.end = time.time()
        self.logger.log_timer(self)

    def __enter__(self):
        self.on_enter()
        return self

    def __exit__(
        self,
        exc_type: "Type[BaseException]" or None = None,
        exc_val: BaseException or None = None,
        exc_tb: TracebackType or None = None,
    ):
        self.on_exit()


class PerIterTimer(Timer):
    n_iters = None

    def set_n_iters(self, n_iters):
        self.n_iters = n_iters

    def on_exit(self):
        super().on_exit()
        if self.n_iters is not None:
            self.logger.log(
                {f"time/{self.get_name()} per iter": self.delta / self.n_iters}
            )


class MethodDecorator(DecoratorInterface, abc.ABC):
    """
    Class intended as base for class method decorators
    to provide the decorator with the instance whose
    method is being called and with the name of the method.
    """

    instance = None
    owner_class: type or None = None
    method_name: str or None = None

    def __call__(self, unbound_method: Callable) -> Callable:
        """Temporarily replace the unbound_method with self."""
        self.function = unbound_method
        return self

    def __set_name__(self, owner_class: type, method_name: str):
        """
        This method is called when owning class owner_class is created,
        here I replace the original unbound method with a wrapper
        that will be given an instance – self, do something with it in self.decorate."""
        self.owner_class = owner_class
        self.method_name = method_name

        @functools.wraps(self.function)
        def method(instance, *args, **kwargs):
            self.instance = instance
            function = self.decorate(
                function=functools.partial(self.function, instance)
            )
            return function(*args, **kwargs)

        setattr(owner_class, method_name, method)


class MethodsDecoratorClassDecorator:
    """Decorator for classes to decorate all user defined methods with decorator."""

    def __init__(self, decorator: Callable):
        self.decorator = decorator

    @staticmethod
    def is_protected_or_builtin_name(attribute_name: str) -> bool:
        return attribute_name.startswith("_")

    @staticmethod
    def is_function(obj) -> bool:
        return isinstance(obj, FunctionType)

    @staticmethod
    def is_functions_first_parameter_self(function: FunctionType) -> bool:
        signature = inspect.signature(function)
        return next(iter(signature.parameters), None) == "self"

    @staticmethod
    def is_unbound_method_defined_in_class(
        unbound_method: FunctionType, cls: type
    ) -> bool:
        return unbound_method.__qualname__.startswith(cls.__name__ + ".")

    def __call__(self, cls):
        for key in dir(cls):
            if self.is_protected_or_builtin_name(key):
                continue
            value = getattr(cls, key)
            if (
                self.is_function(obj=value)
                and self.is_functions_first_parameter_self(function=value)
                and self.is_unbound_method_defined_in_class(
                    unbound_method=value, cls=cls
                )
            ):
                setattr(cls, key, self.decorator(value))
        return cls


class TimerClassDecorator(MethodsDecoratorClassDecorator):
    def __init__(self):
        super().__init__(decorator=Timer())


def download_data_if_needed(config: dict) -> None:
    """
    If config dict has a key "data", then its value be in the following format:
    data:
      - artifact: entity/project_A/dataset_A:latest
        directory: local/data
        match_directory_exactly: True
      - artifact: entity/project_B/model_C:latest
        directory: .
        create_artifact_if_it_doesnt_exist: True
      ...
    Any missing files in directory that are part of the artifact will be downloaded.

    If create_artifact_if_it_doesnt_exist is True and directory exists, then an artifact will be
    created from the directory.

    If match_directory_exactly is True, then content of directory will also be checksummed against the
    artifact manifest and redownloaded if it doesn't exactly match the expected content.
    WARNING: This will DELETE all files in directory that are not included in the artifact.
    """
    if (data_config := config.get("data")) is None:
        return

    for artifact_config in data_config:
        full_artifact_name = artifact_config["artifact"]
        directory = artifact_config["directory"]
        match_directory_exactly = artifact_config.get("match_directory_exactly", False)
        create_artifact_if_it_doesnt_exist = artifact_config.get(
            "create_artifact_if_it_doesnt_exist", False
        )

        if match_directory_exactly and directory == ".":
            raise ValueError(
                f"Passing match_directory_exactly=True and directory='.' will delete "
                f"the files in working directory {os.getcwd()}"
            )
        if create_artifact_if_it_doesnt_exist and not os.path.exists(directory):
            raise ValueError(
                f"Passing create_artifact_if_it_doesnt_exist=True "
                f"requires that directory {directory} exists."
            )

        is_newly_created_artifact = False
        try:
            artifact: wandb.Artifact = wandb.run.use_artifact(full_artifact_name)
        except wandb.errors.CommError as e:
            if not create_artifact_if_it_doesnt_exist:
                raise e
            artifact_name = full_artifact_name.split("/")[-1].split(":")[0]
            artifact = wandb.log_artifact(
                artifact_or_path=directory, name=artifact_name, type="data"
            )
            is_newly_created_artifact = True
        else:
            artifact.download(root=directory)

        if match_directory_exactly and not is_newly_created_artifact:
            try:
                artifact.verify(root=directory)
            except ValueError:
                artifact.checkout(root=directory)
                artifact.verify(root=directory)


def construct_decorator(pass_function_args: bool = False):
    """
    A helper decorator to convert generator with contextlib.contextmanager like
    protocol to a decorator.

    :param pass_function_args: whether to pass function_args
        and function_kwargs keyword arguments to the generator.

    The example of generator with expected protocol:

    @construct_decorator()
    def decorator(function, log_time=True):
        logger = Logger()
        function_return_value = yield
        logger.log(function_return_value)
        if log_time:
            logger.log(time.time())

    @decorator(log_time=False)
    def hello(world=42):
        return world

    """

    def generator_decorator(generator):
        @functools.wraps(generator)
        def wrapped_generator(*generator_args, **generator_kwargs):
            def function_decorator(function):
                @functools.wraps(function)
                def wrapped_function(*function_args, **function_kwargs):
                    if pass_function_args:
                        generator_kwargs.update(
                            dict(
                                function_args=function_args,
                                function_kwargs=function_kwargs,
                            )
                        )
                    iterator = generator(function, *generator_args, **generator_kwargs)
                    next(iterator)
                    result = function(*function_args, **function_kwargs)
                    try:
                        iterator.send(result)
                    except StopIteration:
                        pass
                    return result

                return wrapped_function

            return function_decorator

        return wrapped_generator

    return generator_decorator


@construct_decorator()
def profile_to_file(function: FunctionType, out_file: io.TextIOBase = sys.stderr):
    profiler = line_profiler.LineProfiler()
    profiler.enable_by_count()
    profiler.add_function(function)
    yield
    profiler.print_stats(stream=out_file)


@construct_decorator()
def profile(function: FunctionType, logger: Logger = WandbLogger(), log_every=1):
    logger.register_function(function=function)
    yield
    logger.register_function_call(function=function, log_every=log_every)


def init_torch(debug: bool = True):
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.cuda.init()
        torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(mode=debug)

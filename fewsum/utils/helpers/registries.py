from torch.nn.modules import Module
from mltoolkit.mlmo.utils.tools import BaseConfig
from functools import partial

MODEL_REGISTRY = {}
HP_CONFIG_REGISTRY = {}
RUN_CONFIG_REGISTRY = {}


def _register(cls, name, registry, basecls=None):
    if name in registry:
        raise ValueError("Cannot register duplicate ({})".format(name))
    if basecls is not None and not issubclass(cls, basecls):
        raise ValueError(
            "Model ({}: {}) must extend {}".format(name, cls.__name__,
                                                   basecls.__name__))
    registry[name] = cls
    return cls


def register_model(name):
    """Registers a model."""
    basecls = Module
    registry = MODEL_REGISTRY
    reg_func = partial(_register, name=name, basecls=basecls, registry=registry)
    return reg_func


def register_hp_config(name):
    """Registers a hyper-parameters configuration class."""
    basecls = BaseConfig
    registry = HP_CONFIG_REGISTRY
    reg_func = partial(_register, name=name, basecls=basecls, registry=registry)
    return reg_func


def register_run_config(name):
    """Registers a run configuration class."""
    basecls = BaseConfig
    registry = RUN_CONFIG_REGISTRY
    reg_func = partial(_register, name=name, basecls=basecls, registry=registry)
    return reg_func

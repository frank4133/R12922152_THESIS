import importlib
from models.base_model import BaseModel
import torch.nn as nn


def find_network_using_name(network_name):
    """Import the module "models/[network_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    network_filename = "models." + network_name
    modellib = importlib.import_module(network_filename)
    network = None
    target_model_name = network_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            network = cls

    if network is None:
        raise NotImplementedError("In %s.py, there should be a subclass of "
                            "BaseModel with class name that matches %s in "
                            "lowercase." % (network_filename, target_model_name))
    return network


def create_model(opt):
    """Create a network given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> network = create_model(opt)
    """
    network_class = find_network_using_name(opt.network)
    network_instance = network_class(opt)
    print("network [%s] was created" % type(network_instance).__name__)
    instance = BaseModel(opt, network_instance)
    return instance
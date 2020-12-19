from torch.nn.parallel import DataParallel
import numpy as np


def parallelize_submodules(parent_module):
    """Wraps a parent's sub-modules with a DataParallel object that permits multi
    GPU usage.
    """
    for mod_name, module in parent_module.named_children():
        setattr(parent_module, mod_name, DataParallel(module))


def count_trainable_parameters(module):
    param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return param_count


def count_parameters(module):
    param_count = sum(p.numel() for p in module.parameters())
    return param_count


def get_device(module):
    """Returns the device of the module's parameters."""
    device = None
    for param in module.parameters():
        if device is None:
            device = param.device
            continue
        if device != param.device:
            raise ValueError("Parameters are allocated on different devices.")
    if device is None:
        raise ValueError("No parameters are available in the module.")
    return device


def yield_parameters(module, whitelist=None, blacklist=None, recurse=True):
    """Yields parameters from a module (optionally by a recursive traversal).
    Based on prefix matching. E.g., if whitelist has 'ffnn', it will match
    'ffnn.weight' and 'ffnn.bias'.

    Args:
        module: self-explanatory.
        whitelist: list of parameter names that should be yield.
        blacklist: list of parameter names to be excluded from yield.
        recurse: whether to recursively traverse sub-modules.

    Returns:
        generator with parameters
    """
    if whitelist is not None and blacklist is not None:
        raise ValueError("Both `whitelist` and `blacklist` can't be passed at "
                         "the same time.")
    whitelist = whitelist if whitelist is not None else []
    blacklist = blacklist if blacklist is not None else []
    for name, param in module.named_parameters(recurse=recurse):
        if len(whitelist) and not any([name.startswith(n) for n in whitelist]):
            continue
        if len(blacklist) and any([name.startswith(n) for n in blacklist]):
            continue
        yield param


def unfreeze_parameters(module, whitelist=None, blacklist=None, recurse=True):
    """Unfreezes parameters of a module (optionally by a recursive traversal).

    Args:
        module: self-explanatory.
        recurse: whether to recursively affect sub-modules.
        whitelist: list of parameter names only to be affected by unfreeze.
        blacklist: list of parameter names to be excluded by unfreeze.

    Returns:
        None.
    """
    for param in yield_parameters(module, whitelist=whitelist,
                                  blacklist=blacklist, recurse=recurse):
        param.requires_grad = True


def freeze_parameters(module, whitelist=None, blacklist=None, recurse=True):
    """Unfreezes parameters of a module (optionally by a recursive traversal).

    Args:
        module: self-explanatory.
        recurse: whether to recursively affect sub-modules.
        whitelist: list of parameter names only to be affected by freeze.
        blacklist: list of parameter names to be excluded by freeze.

    Returns:
        None.
    """
    for param in yield_parameters(module, whitelist=whitelist,
                                  blacklist=blacklist, recurse=recurse):
        param.requires_grad = False

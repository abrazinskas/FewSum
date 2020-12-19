import torch as T
from torch.nn import DataParallel, Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.data_parallel import _check_balance
from torch.nn.parallel.replicate import replicate
from torch import Tensor
import inspect
from mltoolkit.mlmo.utils.tools import DecState


class ModuleParallel(Module):
    """Implements data parallelism at the module level that has custom output.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass,
    gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    The difference from the standard `DataParallel` class is that it parallelizes
    not only `forward` but also other public methods of a module object.
    Assumes that each method yields dicts with statistics and tensors.
    The former are summed and divided by the number of replicas, the latter are
    concatenated by the 0-dimension.
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ModuleParallel, self).__init__()

        if not T.cuda.is_available():
            self.module = module
            self.device_ids = []
        else:
            if device_ids is None:
                device_ids = list(range(T.cuda.device_count()))
            if output_device is None:
                output_device = device_ids[0]

            self.dim = dim
            self.module = module
            self.device_ids = list(map(lambda x: _get_device_index(x, True),
                                       device_ids))
            self.output_device = _get_device_index(output_device, True)

            _check_balance(self.device_ids)

            if len(self.device_ids) == 1:
                self.module.cuda(device_ids[0])

        self._def_methods = {mn for mn, m
                             in inspect.getmembers(Module(), inspect.ismethod)}
        self._def_methods.remove('forward')
        self.make_module_methods_parallel()

        def_attrs = {k for k, v in Module().__dict__.items()
                     if not inspect.ismethod(v)}
        for k, v in module.__dict__.items():
            if k is not inspect.ismethod(v) and k not in def_attrs:
                setattr(self, k, v)

    def gather(self, outputs, output_device):
        """Gathers outputs from replicas, aggregates them generically.

        Gathers outputs from replicas, assumes that each element of the output
        is a PyTorch tensor(1+dim), DecStates, or dictionary of scalars.
        The first and second are concatenated along 0-dim, the last are
        added and then divided by the number of replicas (mean of means).

        It does not check that the outputs are consistent across replicas. At
        the moment.

        Args:
            outputs: replica outputs.
            output_device:

        Returns:
            aggregated output.
        """
        def _coll_tensor(coll_list, out_tensor):
            out_tensor = gather(outputs=[out_tensor],
                                target_device=self.output_device, dim=0)
            coll_list.append(out_tensor)

        def _coll_dict(coll_dict, out_dict):
            for k, v in out_dict.items():
                if k not in coll_dict:
                    coll_dict[k] = []
                coll_dict[k].append(v)

        def _coll_dec_state(coll_dstate, out_dstate):
            for attr_name, attr_val in out_dstate.__dict__.items():
                if isinstance(attr_val, Tensor):
                    if getattr(coll_dstate, attr_name) is None:
                        setattr(coll_dstate, attr_name, [])
                    _coll_tensor(getattr(coll_dstate, attr_name), attr_val)
                elif isinstance(attr_val, dict):
                    if getattr(coll_dstate, attr_name) is None:
                        setattr(coll_dstate, attr_name, dict())
                    _coll_dict(getattr(coll_dstate, attr_name), attr_val)
                elif attr_val is None:
                    continue
                else:
                    raise NotImplementedError

        # minor reformatting
        for indx, o in enumerate(outputs):
            if not isinstance(o, (list, tuple)):
                outputs[indx] = [o]

        coll_outputs = [None for _ in range(len(outputs[0]))]

        # collecting outputs
        for output in outputs:
            for indx, o in enumerate(output):
                # statistics, such as loss, assumed to be of simple types
                if isinstance(o, dict):
                    if coll_outputs[indx] is None:
                        coll_outputs[indx] = dict()
                    _coll_dict(coll_outputs[indx], o)
                # tensors
                elif isinstance(o, Tensor):
                    if coll_outputs[indx] is None:
                        coll_outputs[indx] = []
                    _coll_tensor(coll_outputs[indx], o)
                # decoder state
                elif isinstance(o, DecState):
                    if coll_outputs[indx] is None:
                        coll_outputs[indx] = DecState()
                    coll_dec_state = coll_outputs[indx]
                    _coll_dec_state(coll_dec_state, o)
                else:
                    raise NotImplementedError

        # aggregating outputs
        for indx, o in enumerate(coll_outputs):
            if isinstance(o, list):
                coll_outputs[indx] = T.cat(o)
            elif isinstance(o, dict):
                coll_outputs[indx] = {k: sum(v) / len(v) for k, v in o.items()}
            elif isinstance(o, DecState):
                for attr_name, attr_val in o.__dict__.items():
                    if isinstance(attr_val, list):
                        setattr(o, attr_name, T.cat(attr_val))
                    elif isinstance(attr_val, dict):
                        for k, v in attr_val.items():
                            attr_val[k] = T.cat(v)
                    elif attr_val is None:
                        continue
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError

        if len(coll_outputs) == 1:
            coll_outputs = coll_outputs[0]

        return coll_outputs

    def replicate(self, module, device_ids):
        modules = replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def parallel_decorator(self, method_name):
        """Creates a decorator to execute a method on multiple replicas."""
        def _run(*inputs, **kwargs):
            if not self.device_ids:
                return getattr(self.module, method_name)(*inputs, **kwargs)
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                return getattr(self.module, method_name)(*inputs[0], **kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            rep_methods = [getattr(rep, method_name) for rep in replicas]
            outputs = self.parallel_apply(rep_methods, inputs, kwargs)
            return self.gather(outputs, self.output_device)
        return _run

    def make_module_methods_parallel(self):
        method_names = inspect.getmembers(self.module, inspect.ismethod)
        for method_name, _ in method_names:
            if self.allowed_method(method_name):
                setattr(self, method_name, self.parallel_decorator(method_name))
        
    def allowed_method(self, name):
        """Excludes inherited Module and protected methods."""
        if name in self._def_methods:
            return False
        if name[0] == "_":
            return False
        return True


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each
    module created by original replication.

    The callback will be invoked with arguments
    `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a
    context (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be
    called ahead of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class CallbackContext(object):
    pass


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, T._six.string_classes):
        device = T.device(device)
    if isinstance(device, T.device):
        dev_type = device.type
        if device.type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return T.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with a specified index '
                             'or an integer, but got: '.format(device))
    return device_idx

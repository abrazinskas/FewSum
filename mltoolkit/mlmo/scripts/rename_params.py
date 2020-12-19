import argparse
import torch as T
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS
from mltoolkit.mlutils.helpers.argparse import str2list, str2bool


def rename_params(input_fp, output_fp, old_attr_names, new_attr_names):
    """Renames a model's parameters, saves them to an output file."""
    assert len(old_attr_names) == len(new_attr_names)
    model_params = T.load(input_fp, 'cpu')[MODEL_PARAMS]
    for old_name, new_name in zip(old_attr_names, new_attr_names):
        model_params[new_name] = model_params[old_name]
        del model_params[old_name]
    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: model_params}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fp", type=str)
    parser.add_argument("--output-fp", type=str)
    parser.add_argument("--old-attr-names", type=str2list)
    parser.add_argument("--new-attr-names", type=str2list)
    args = parser.parse_args()
    rename_params(**vars(args))

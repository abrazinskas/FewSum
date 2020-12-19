import argparse
import torch as T
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS


def extract_params(input_fp, output_fp, attr_names, device='cpu'):
    """Extract a subset parameters from the file. Saves to a new file."""
    model_params = T.load(input_fp, device)[MODEL_PARAMS]
    params_to_save = {}
    for attr_name in attr_names:
        params_to_save[attr_name] = model_params[attr_name]

    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: params_to_save}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--attr_names", nargs='+')

    args = parser.parse_args()
    extract_params(**vars(args))

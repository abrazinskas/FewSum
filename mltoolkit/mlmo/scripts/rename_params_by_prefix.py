import argparse
import torch as T
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS
from mltoolkit.mlutils.helpers.argparse import str2list


def rename_params_by_prefix(input_fp, output_fp, old_prefixes, new_prefixes):
    """Renames a model's parameters by matching prefixes, saves them to an
    output file. Renaming is performed in multiple iterations if multiple
    prefixes are given.
    """
    assert len(old_prefixes) == len(new_prefixes)
    model_params = T.load(input_fp, 'cpu')[MODEL_PARAMS]
    for old_prefix, new_prefix in zip(old_prefixes, new_prefixes):
        tmp_params = dict()
        for curr_name, param in model_params.items():
            if curr_name.startswith(old_prefix):
                new_name = new_prefix + curr_name[len(old_prefix):]
                print(f"{curr_name} => {new_name}")
                curr_name = new_name
            tmp_params[curr_name] = param
        model_params = tmp_params
    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: model_params}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fp", type=str)
    parser.add_argument("--output-fp", type=str)
    parser.add_argument("--old-prefixes", type=str2list)
    parser.add_argument("--new-prefixes", type=str2list)
    rename_params_by_prefix(**vars(parser.parse_args()))

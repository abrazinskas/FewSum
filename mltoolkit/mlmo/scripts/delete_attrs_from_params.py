import argparse
import torch as T
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlutils.helpers.logging_funcs import init_logger
from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS
from mltoolkit.mlutils.helpers.argparse import str2list, str2bool
import re

logger = init_logger(" ")


def delete_attr_from_params(input_fp, output_fp, attr_names, match_start=False):
    """Removes a particular attrs from the dictionary of params, saves back.

    Args:
        input_fp:
        output_fp:
        attr_names:
        match_start: if set to True will match based on the beginning of the
            string. E.g., _encoder match _encoder.param1.linear.weights.
    """
    # TODO: explain how regex works
    model_params = T.load(input_fp, 'cpu')[MODEL_PARAMS]
    if match_start:
        for attr_name in attr_names:
            r_aname = re.compile("^"+attr_name)
            for param_name in list(model_params.keys()):
                if r_aname.match(param_name):
                    del model_params[param_name]
                    logger.info("Deleting: %s." % param_name)
    else:
        for attr_name in attr_names:
            if attr_name in model_params:
                del model_params[attr_name]
                logger.info("Deleting: %s." % attr_name)

    # dumping to the disk
    safe_mkfdir(output_fp)
    T.save({MODEL_PARAMS: model_params}, f=output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--attr_names", type=str2list)
    parser.add_argument("--match_start", type=str2bool, default=False)

    args = parser.parse_args()
    delete_attr_from_params(**vars(args))

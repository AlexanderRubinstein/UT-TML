import os
import traceback
import pickle
from hashlib import blake2b
import numpy as np
from datetime import datetime
import torch
import random


DEFAULT_HASH_SIZE = 10


def raise_unknown(param, value, location=""):
    exception_msg = "Unknown {}".format(param)
    if location:
        exception_msg += " in {}".format(location)
    exception_msg += ": {}".format(value)
    raise Exception(exception_msg)


def default_save_func(object_to_save, path):
    pickle.dump(object_to_save, open(path, "wb"))


def default_load_func(path):
    return pickle.load(open(path, "rb"))


def make_or_load_from_cache(
    object_name,
    object_config,
    make_func,
    save_func=default_save_func,
    load_func=default_load_func,
    cache_path=None,
    forward_cache_path=False,
    logger=None
):

    if cache_path is None:
        cache_fullpath = None
    else:
        os.makedirs(cache_path, exist_ok=True)
        cache_fullpath = os.path.join(
            cache_path,
            "{}_{}.pkl".format(
                object_name,
                get_hash(object_config)
            )
        )
    if cache_fullpath and os.path.exists(cache_fullpath):
        log_or_print(
            "Loading cached {} from {}".format(
                object_name,
                cache_fullpath
            ),
            logger=logger
        )
        result = load_func(cache_fullpath)
    else:
        if forward_cache_path:
            result = make_func(
                object_config,
                cache_path=cache_path
            )
        else:
            result = make_func(
                object_config
            )
        if cache_fullpath:
            try:
                save_func(result, cache_fullpath)
                log_or_print(
                    "Saved cached {} into {}".format(
                        object_name,
                        cache_fullpath
                    ),
                    logger=logger
                )
            except OSError as err:
                log_or_print(
                    "Could not save cached {} to {}. "
                    "Reason: \n{} \nContinuing without saving it.".format(
                        object_name,
                        cache_fullpath,
                        traceback.format_exc()
                    ),
                    logger=logger
                )
    return result


def log_or_print(msg, logger, msg_type="log"):
    if logger:
        if msg_type == "log":
            logger.log(msg)
        elif msg_type == "error":
            logger.error(msg)
        elif msg_type == "info":
            logger.info(msg)
        else:
            raise_unknown("msg_type", msg_type, "log_or_pring")
    else:
        print(msg)


def get_hash(input_object, hash_size=DEFAULT_HASH_SIZE):
    h = blake2b(digest_size=hash_size)
    h.update(input_object.__repr__().encode())
    return h.hexdigest()


def read_dsprites_npz(filename):
    return np.load(filename, allow_pickle=True, encoding='latin1')


def get_current_time():
    return datetime.now()


def compute_proportion(proportion, total_number):
    return max(1, int(
        proportion * total_number
    ))


def apply_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def deterministically_subsample_indices_uniformly(
    total_samples,
    num_to_subsample
):
    return torch.linspace(
        0,
        total_samples - 1,
        num_to_subsample,
        dtype=torch.int
    )

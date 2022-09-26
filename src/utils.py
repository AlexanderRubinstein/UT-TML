import os
import traceback
import pickle
from hashlib import blake2b
import numpy as np
from datetime import datetime
import torch
import random
import matplotlib.pyplot as plt


DEFAULT_HASH_SIZE = 10
PLT_ROW_SIZE = 4
PLT_COL_SIZE = 2


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


def show_image(image):
    image = image.squeeze()
    plt.imshow(image, cmap=get_cmap(image))
    plt.show(block=True)


def get_cmap(image):
    cmap = "viridis"
    if len(image.shape) == 2:
        cmap = "gray"
    return cmap


def show_images_batch(images_batch, labels_batch=None):
    images_list = []
    labels_list = None if labels_batch is None else []
    n_images = images_batch.shape[0]
    if labels_batch is not None:
        assert labels_batch.shape[0] == n_images
    for i in range(n_images):
        images_list.append(images_batch[i].squeeze())
        if labels_list is not None:
            labels_list.append(labels_batch[i].item())
    show_images(images_list, labels_list)


def show_images(images, labels=None):

    def remove_ticks_and_labels(subplot):
        subplot.axes.xaxis.set_ticklabels([])
        subplot.axes.yaxis.set_ticklabels([])
        subplot.axes.xaxis.set_visible(False)
        subplot.axes.yaxis.set_visible(False)

    def get_row_cols(n):
        n_rows = int(np.sqrt(n))
        n_cols = int(n / n_rows)
        if n % n_rows != 0:
            n_cols += 1
        return n_rows, n_cols

    n = len(images)
    assert n > 0
    if labels is not None:
        assert n == len(labels)

    n_rows, n_cols = get_row_cols(n)

    cmap = get_cmap(images[0])
    fig = plt.figure(figsize=(n_rows * PLT_ROW_SIZE, n_cols * PLT_COL_SIZE))
    for i in range(n):
        subplot = fig.add_subplot(n_rows, n_cols, i + 1)
        title = f'{i}'
        if labels is not None:
            title += f" of label \"{labels[i]}\""
        subplot.title.set_text(title)
        remove_ticks_and_labels(subplot)

        subplot.imshow(images[i], cmap=cmap)

    plt.show(block=True)

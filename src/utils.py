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
PLT_COL_SIZE = 4


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


# def read_dsprites_npz(filename):
#     return np.load(filename, allow_pickle=True, encoding='latin1')


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


def imshow(plot, image, cmap=None):
    image = image.squeeze()
    num_image_dims = len(image.shape)
    if cmap is None:
        cmap = get_cmap(image)
    assert num_image_dims >= 2 and num_image_dims <= 3
    if num_image_dims == 3:
        # image = image.transpose((1, 2, 0))
        image = np.transpose(image, (1, 2, 0))
    plot.imshow(image, cmap=cmap)

def show_image(image):
    # image = image.squeeze()
    # plt.imshow(image, cmap=get_cmap(image))
    imshow(plt, image)
    plt.show(block=True)


def get_cmap(image):
    cmap = "viridis"
    squeezed_shape = image.squeeze().shape
    if len(squeezed_shape) == 2:
        cmap = "gray"
    return cmap


def append_to_list_in_dict(d, key, element):
    if key in d:
        assert isinstance(d[key], list)
        d[key].append(element)
    else:
        d[key] = [element]


def show_images_batch(images_batch, label_batches=None):
    images_list = []

    images_batch = images_batch.cpu()

    # labels_list = None if label_batches is None else []
    label_lists = None if label_batches is None else {}
    n_images = images_batch.shape[0]
    if label_batches is not None:
        if not isinstance(label_batches, dict):
            label_batches = {"label": label_batches}
        # for label_name, label_batch in label_batches.items():
        for label_batch in label_batches.values():
            assert label_batch.shape[0] == n_images
            # label_lists[label_name] = []
    for i in range(n_images):
        # images_list.append(images_batch[i].squeeze())
        images_list.append(images_batch[i])
        if label_lists is not None:
            # label_lists.append(labels_batch[i].item())
            for label_name, label_batch in label_batches.items():
                append_to_list_in_dict(label_lists, label_name, label_batch[i].item())
    # show_images(images_list, labels_list)
    show_images(images_list, label_lists)


# def show_images(images, labels=None):
def show_images(images, label_lists=None):

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
    if label_lists is not None:
        # assert n == len(labels)
        for label_list in label_lists.values():
            assert len(label_list) == n

    n_rows, n_cols = get_row_cols(n)

    cmap = get_cmap(images[0])
    fig = plt.figure(figsize=(n_cols * PLT_COL_SIZE, n_rows * PLT_ROW_SIZE))
    for i in range(n):
        subplot = fig.add_subplot(n_rows, n_cols, i + 1)
        title = f'n{i}'
        if label_lists is not None:
            for label_name, label_list in label_lists.items():
                title += f"\n{label_name}=\"{label_list[i]}\""
        subplot.title.set_text(title)
        remove_ticks_and_labels(subplot)

        # subplot.imshow(images[i], cmap=cmap)
        imshow(subplot, images[i], cmap=cmap)

    plt.tight_layout()
    plt.show(block=True)


def append_dict(total_dict, current_dict):
    """
    Append leaves of possibly nested <current_dict>
    to leaf lists of possibly nested <total_dict>
    """

    is_new_total_dict = False
    if len(total_dict) == 0:
        is_new_total_dict = True
    for key, value in current_dict.items():
        if isinstance(value, dict):
            if is_new_total_dict:
                sub_dict = {}
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
            else:
                assert key in total_dict
                sub_dict = total_dict[key]
                assert isinstance(sub_dict, dict)
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
        else:
            if is_new_total_dict:
                total_dict[key] = [value]
            else:
                assert key in total_dict
                assert isinstance(total_dict[key], list)
                total_dict[key].append(value)


def subsample_list_by_indices(input_list, num_to_subsample):
    # subsampled_indices = utils.deterministically_subsample_indices_uniformly(
    subsampled_indices = deterministically_subsample_indices_uniformly(
        len(input_list),
        num_to_subsample
    ).numpy()
    return np.array(input_list)[subsampled_indices].tolist()


PLT_PLOT_HEIGHT = 5
PLT_PLOT_WIDTH = 5


def plot_stats(stats, title):

    n = len(stats)
    fig = plt.figure(figsize=(PLT_PLOT_WIDTH, n * PLT_PLOT_HEIGHT))

    for i, (stat_name, stat_history) in enumerate(stats.items()):
        subplot = fig.add_subplot(n, 1, i + 1)
        title = f'{stat_name}'

        subplot.title.set_text(title)

        subplot.plot(stat_history)

    plt.show(block=True)


def show_dataloader(dataloader, label_names):
    images_batch, labels_batch = next(iter(dataloader))

    images_batch = images_batch.cpu()
    if isinstance(labels_batch, list):
        for i in range(len(labels_batch)):
            labels_batch[i] = labels_batch[i].cpu()
    else:
        labels_batch = labels_batch.cpu()

    # if isinstance(labels_batch, list):
    #     labels_batch = {
    #         cue: label_batch for cue, label_batch in zip(label_names, labels_batch)
    #     }
    # else:
    #     assert len(label_names) == 1
    #     labels_batch = {label_names[0]: labels_batch}

    labels_batch = make_named_labels_batch(label_names, labels_batch)

    show_images_batch(images_batch, labels_batch)


def make_named_labels_batch(label_names, labels_batch):
    if isinstance(labels_batch, list):
        labels_batch = {
            cue: label_batch for cue, label_batch in zip(label_names, labels_batch)
        }
    else:
        assert len(label_names) == 1
        labels_batch = {label_names[0]: labels_batch}

    return labels_batch

import os
import traceback
import pickle
from hashlib import blake2b
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import gc
from collections import UserDict
import warnings
from typing import (
    Union,
    Dict,
    List,
    Any,
    Callable
)


DEFAULT_HASH_SIZE = 10
PLT_ROW_SIZE = 4
PLT_COL_SIZE = 4
PLT_PLOT_HEIGHT = 5
PLT_PLOT_WIDTH = 5


FINGERPRINT_ATTR = "_object_fingerprint_for_reading_from_cache"


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
    object_name: str,
    object_config: Dict[str, Any],
    make_func: Callable,
    save_func: Callable = default_save_func,
    load_func: Callable = default_load_func,
    cache_path: str = None,
    forward_cache_path: bool = False
):
    """
    Make a resulting object or load it from cache (either in RAM or filesystem)
    if it has been created and saved by this function before.

    Args:

        object_name (str): a name of the object to make or load from cache;
            used for making its unique cache name.

        object_config (Dict[str, Any]): a config used by the <make_func>;
            its hash also used to make the resulting object's unique cache name.

        make_func (Callable): an object factory that makes resulting object
            in case it is not found in cache.
            Uses <object_config> and optionally <cache_path>,
            if <forward_cache_path> is True, as arguments.

        save_func (Callable): a function used to save object in the filesystem.
            Default: default_save_func

        load_func (Callable): a function used to load resulting object
            from the cache in the filesystem.
            Default: default_load_func

        cache_path (str): a path for saving cache in the filesystem.
            If is None, cache is not saved in the filesystem.
            Default: None

        forward_cache_path (bool): a flag, if it is True,
            <cache_path> is forwarded as an argument to the <make_func>.
            Default: False

    Returns:

        result (Any): the resulting object.
    """


    def update_object_fingerprint_attr(result, object_fingerprint):

        if isinstance(result, dict):
            result = UserDict(result)

        setattr(result, FINGERPRINT_ATTR, object_fingerprint)
        return result


    object_fingerprint = "{}_{}".format(object_name, get_hash(object_config))

    objects_with_the_same_fingerprint = extract_from_gc_by_attribute(
        FINGERPRINT_ATTR,
        object_fingerprint
    )
    if len(objects_with_the_same_fingerprint) > 0:
        print(
            "Reusing object from RAM with fingerprint {}".format(
                object_fingerprint
            )
        )
        return objects_with_the_same_fingerprint[0]

    if cache_path is None:
        cache_fullpath = None
    else:
        os.makedirs(cache_path, exist_ok=True)
        cache_fullpath = os.path.join(
            cache_path,
            "{}.pkl".format(object_fingerprint)
        )
    if cache_fullpath and os.path.exists(cache_fullpath):

        print(
            "Loading cached {} from {}".format(
                object_name,
                cache_fullpath
            )
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
                print(
                    "Saved cached {} into {}".format(
                        object_name,
                        cache_fullpath
                    )
                )

            except OSError as err:
                print(
                    "Could not save cached {} to {}. "
                    "Reason: \n{} \nContinuing without saving it.".format(
                        object_name,
                        cache_fullpath,
                        traceback.format_exc()
                    )
                )

    result = update_object_fingerprint_attr(result, object_fingerprint)


    return result


def has_nested_attr(object, nested_attr):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return hasattr(object, nested_attr[0])
    else:
        return (
            hasattr(object, nested_attr[0])
                and has_nested_attr(
                    getattr(object, nested_attr[0]),
                    nested_attr[1:]
                )
        )


def get_nested_attr(object, nested_attr):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return getattr(object, nested_attr[0])
    else:
        return (
            get_nested_attr(getattr(object, nested_attr[0]), nested_attr[1:])
        )


def set_nested_attr(object, nested_attr, value):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return setattr(object, nested_attr[0], value)
    else:
        set_nested_attr(getattr(object, nested_attr[0]), nested_attr[1:], value)


def extract_from_gc_by_attribute(attribute_name, attribute_value):

    res = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for obj in gc.get_objects():
            has_attribute = False

            try:
                has_attribute = hasattr(obj, attribute_name)
            except:
                continue

            if (
                has_attribute
                    and (getattr(obj, attribute_name) == attribute_value)
            ):
                res.append(obj)

    return res


def get_hash(input_object, hash_size=DEFAULT_HASH_SIZE):
    h = blake2b(digest_size=hash_size)
    h.update(input_object.__repr__().encode())
    return h.hexdigest()


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


def imshow(plot, image, cmap=None, color_dim_first=True):
    image = image.squeeze()
    num_image_dims = len(image.shape)
    if cmap is None:
        cmap = get_cmap(image)
    assert num_image_dims == 2 or num_image_dims == 3
    if num_image_dims == 3 and color_dim_first:
        image = np.transpose(image, (1, 2, 0))
    plot.imshow(image, cmap=cmap)


def show_image(image, color_dim_first=True):
    imshow(plt, image, color_dim_first=color_dim_first)
    plt.show()


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


def show_images_batch(
    images_batch: torch.tensor,
    label_batches: Union[torch.tensor, Dict[str, torch.tensor]] = None
):
    """
    Shows a batch of images as a square image grid.
    If <label_batches> is provided, each image title consists of label names
    and their corresponding values.

    Args:

        images_batch (torch.tensor): of images batch.

        label_batches (Union[torch.tensor, Dict[str, torch.tensor]], optional):
            information about labels that is either a label batch
            or a dict that maps label name to the corresponding label batch.
    """


    images_list = []

    images_batch = images_batch.cpu()

    label_lists = None if label_batches is None else {}

    n_images = images_batch.shape[0]

    if label_batches is not None:

        if not isinstance(label_batches, dict):
            label_batches = {"label": label_batches}

        for label_batch in label_batches.values():
            assert label_batch.shape[0] == n_images

    for i in range(n_images):
        images_list.append(images_batch[i])
        if label_lists is not None:
            for label_name, label_batch in label_batches.items():
                append_to_list_in_dict(
                    label_lists,
                    label_name,
                    label_batch[i].item()
                )

    show_images(images_list, label_lists)


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

        imshow(subplot, images[i], cmap=cmap)

    plt.tight_layout()
    plt.show()


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


def subsample_list_uniformly(input_list, num_to_subsample):

    subsampled_indices = deterministically_subsample_indices_uniformly(
        len(input_list),
        num_to_subsample
    ).numpy()

    return np.array(input_list)[subsampled_indices].tolist()


def plot_stats(stats, title):

    n = len(stats)
    fig = plt.figure(figsize=(PLT_PLOT_WIDTH, n * PLT_PLOT_HEIGHT))

    for i, (stat_name, stat_history) in enumerate(stats.items()):

        subplot = fig.add_subplot(n, 1, i + 1)

        title = str(stat_name)

        subplot.title.set_text(title)

        subplot.plot(stat_history)

    plt.show()


def show_dataloader_first_batch(
    dataloader: torch.utils.data.DataLoader,
    label_names: List[str]
):
    """
    Plots first batch of a dataloader
    using local function "show_images_batch".

    Args:

        dataloader (torch.utils.data.DataLoader): a dataloader
            which first batch is shown.

        label_names (List[str]): list of names for each label.
            For example, if a dataloader generates
            tuple (input, List[label_0, ... label_k]),
            label_names will be List[label_0_name, ..., label_k_name].
    """

    images_batch, labels_batch = next(iter(dataloader))

    images_batch = images_batch.cpu()

    if isinstance(labels_batch, list):
        for i in range(len(labels_batch)):
            labels_batch[i] = labels_batch[i].cpu()
    else:
        labels_batch = labels_batch.cpu()

    labels_batch = make_named_labels_batch(label_names, labels_batch)

    show_images_batch(images_batch, labels_batch)


def make_named_labels_batch(label_names, labels_batch):
    if isinstance(labels_batch, list):
        assert len(labels_batch) == len(label_names)
        labels_batch = {
            label_name: label_batch
                for label_name, label_batch
                    in zip(label_names, labels_batch)
        }
    else:
        assert len(label_names) == 1
        labels_batch = {label_names[0]: labels_batch}

    return labels_batch

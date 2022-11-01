import numpy as np
import torch
import random
from typing import (
    Dict,
    Callable,
    Union,
    Any
)


# local modules
import utils


DSPRITES_NPZ_PATH = "/tmp/dsprites.npz"
DSPRITES_NUM_COLORS = 3
IS_DSPRITES_COLORED = True
N_COLORS = DSPRITES_NUM_COLORS if IS_DSPRITES_COLORED else 1
GROUND_TRUTH_CUE = "scale"
EASY_TO_BIAS_CUE = "color" if IS_DSPRITES_COLORED else "posX"
NUM_CLASSES = 3


EMPTY_LABEL = -1
ERROR_EPS = 1e-3
TRAIN_DATASET_SIZE = 20000
TEST_DATASET_SIZE = 2000
TRAIN_VAL_SPLIT = 0.77
BATCH_SIZE = 64
CACHE_PATH = "/tmp/ood_cache"
DEFAULT_DSPRITES_HOLDER_ARGS = {
    "train_val_split": TRAIN_VAL_SPLIT,
    "colored": IS_DSPRITES_COLORED,
    "cache_path": CACHE_PATH
}


class DSpritesHolder:
    """
    An object that stores images and their factor (aka cue) values
    from the dSprites dataset; each sample belongs to only one
    of two disjoint splits {"train", "test"}
    """
    def __init__(
        self,
        train_val_split: float,
        colored: bool
    ):
        """
        Creates a DSpritesHolder object

        Args:

            train_val_split (float): proportion of samples that
                will belong to the "train" split.
                As a consequence, (1 - train_val_split) is the proportion
                of samples that will belong to the "test" split.

            colored (bool): flag, if it is True, images will be colored.
                When images are not colored they do not have color channel,
                i.e. len(image.shape) == 2 for non-colored images
                and (len(image.shape) == 3 and image.shape[0] == 3) for colored.
        """

        dsprites_zip = read_dsprites_npz(DSPRITES_NPZ_PATH)
        self.imgs = dsprites_zip['imgs']
        metadata = dsprites_zip['metadata'][()]
        self.latents_names = metadata['latents_names']

        self.latents_names_to_indices = {}
        for i, name in enumerate(self.latents_names):
            self.latents_names_to_indices[name] = i

        self.latents_sizes = metadata['latents_sizes']
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,]))
        )

        self.train_val_split = train_val_split

        self.num_images_of_one_color = self.imgs.shape[0]
        self.colored = colored

        if self.colored:
            self.latents_sizes[0] = DSPRITES_NUM_COLORS
            self.colored_image_shape \
                = (self.latents_sizes[0],) \
                    + self.imgs[0].shape

        self.total_num_images \
            = self.num_images_of_one_color * self.latents_sizes[0]

        self.indices = {}
        self.indices["train"], self.indices["test"] \
            = self._split_into_train_and_test()
        self.unique_fingerprint = self._create_holder_fingerprint()

    def _create_holder_fingerprint(self):
        return (
            utils.get_hash(self.imgs)
                + str(self.colored)
                + utils.get_hash(self.indices)
        )

    def _index_to_values_combination(self, index):
        combination = []
        for base in self.latents_bases:
            current_value = int(index / base)
            combination.append(current_value)
            index = index % base
            if len(combination) == len(self.latents_bases):
                assert index == 0
        return combination

    def __getitem__(self, idx):

        if self.colored:

            assert self.colored_image_shape
            assert self.num_images_of_one_color
            color = int(idx / self.num_images_of_one_color)
            original_idx = idx % self.num_images_of_one_color
            colored_image = np.zeros(self.colored_image_shape)
            colored_image[color, ...] = self.imgs[original_idx]
            image = colored_image

        else:

            image = self.imgs[idx][None, ...]

        return image

    def _split_into_train_and_test(self):

        assert self.total_num_images
        all_indices = np.array(range(self.total_num_images))
        train_indices_positions \
            = utils.deterministically_subsample_indices_uniformly(
                self.total_num_images,
                utils.compute_proportion(
                    self.train_val_split,
                    self.total_num_images
                )
            ).numpy()
        train_indices = all_indices[train_indices_positions].tolist()
        test_indices = np.delete(
            all_indices,
            train_indices_positions,
            axis=0
        ).tolist()

        return train_indices, test_indices


def assert_same_dsprites_holders(holder_one, holder_two):
    assert holder_one.unique_fingerprint
    assert holder_two.unique_fingerprint
    assert holder_one.unique_fingerprint == holder_two.unique_fingerprint


def make_dsprites_holder_from_config(dsprites_holder_config):
    return DSpritesHolder(
        train_val_split=dsprites_holder_config["train_val_split"],
        colored=dsprites_holder_config["colored"]
    )


def read_dsprites_npz(filename):
    return np.load(filename, allow_pickle=True, encoding='latin1')


class DatasetLabelledAccordingToCue:

    def __init__(self, dsprites_holder, num_classes, cue, split):

        def distribute_latent_values_to_classes(num_classes, latent_size):
            assert num_classes <= latent_size

            num_per_class = int(latent_size / num_classes)
            remainder = latent_size % num_classes

            upperbound_latent_value_id_per_class = [
                (
                    (class_id + 1)
                        * num_per_class
                        + min(class_id + 1, remainder)
                )
                    for class_id in range(num_classes)
            ]

            assert upperbound_latent_value_id_per_class[-1] == latent_size
            assert len(upperbound_latent_value_id_per_class) == num_classes

            return upperbound_latent_value_id_per_class

        def dict_latent_value_id_to_label(upperbound_latent_value_id_per_class):

            latent_value_id_to_label = {}

            latent_size = upperbound_latent_value_id_per_class[-1]
            class_id = 0

            for latent_value_id in range(latent_size):

                if (
                    latent_value_id
                        >= upperbound_latent_value_id_per_class[class_id]
                ):
                    class_id += 1

                assert (
                    latent_value_id
                        < upperbound_latent_value_id_per_class[class_id]
                )

                latent_value_id_to_label[latent_value_id] = class_id

            return latent_value_id_to_label

        self.dsprites_holder = dsprites_holder
        cue_id = self.dsprites_holder.latents_names_to_indices[cue]
        upperbound_latent_value_id_per_class \
            = distribute_latent_values_to_classes(
                num_classes,
                self.dsprites_holder.latents_sizes[cue_id]
            )
        self.latent_values_to_labels = dict_latent_value_id_to_label(
            upperbound_latent_value_id_per_class
        )
        self.index_label_pairs = []

        for index in self.dsprites_holder.indices[split]:

            latent_values_combination \
                = self.dsprites_holder._index_to_values_combination(index)
            latent_value_id = latent_values_combination[cue_id]
            label = self.latent_values_to_labels[latent_value_id]
            self.index_label_pairs.append((index, label))

    def __getitem__(self, idx):
        assert len(self.index_label_pairs) > 0
        index_in_dsprites_holder, label = self.index_label_pairs[idx]

        return (torch.Tensor(
            self.dsprites_holder[index_in_dsprites_holder]),
            torch.tensor(label, dtype=torch.int64)
        )

    def __len__(self):
        return len(self.index_label_pairs)


def make_dataset_labelled_according_to_cue(dataset_config):
    return DatasetLabelledAccordingToCue(
        dsprites_holder=prepare_dsprites_holder_maker(
            **dataset_config["dsprites_holder_args"]
        )(),
        num_classes=dataset_config["num_classes"],
        cue=dataset_config["cue"],
        split=dataset_config["split"]
    )


class MultilabelDataset:
    def __init__(self, dsprites_holder, num_classes, split):

        image_to_labels = {}
        self.dsprites_holder = dsprites_holder
        self.cue_names = []

        for cue in self.dsprites_holder.latents_names:

            if not IS_DSPRITES_COLORED and cue == "color":
                continue

            self.cue_names.append(cue)

            one_cue_dataset = DatasetLabelledAccordingToCue(
                dsprites_holder=self.dsprites_holder,
                num_classes=num_classes,
                cue=cue,
                split=split
            )

            for image_idx, label in one_cue_dataset.index_label_pairs:
                utils.append_to_list_in_dict(image_to_labels, image_idx, label)

        self.image_to_labels_list = list(image_to_labels.items())

    def __getitem__(self, idx):

        image_idx, labels_list = self.image_to_labels_list[idx]

        return (
            torch.Tensor(self.dsprites_holder[image_idx]),
            [torch.tensor(label, dtype=torch.int64) for label in labels_list]
        )

    def __len__(self):
        return len(self.image_to_labels_list)


class DatasetWithDiagAndOffDiagCells:

    def __init__(
        self,
        dataset_labelled_according_to_easy_to_bias_cue,
        dataset_labelled_according_to_ground_truth_cue,
        off_diag_proportion,
        unlabelled_off_diag,
        equal_diag_off_diag_in_batch,
        shuffle,
        dataset_size,
        off_diag_multiplier
    ):

        def make_labels_to_indices_set(index_label_pairs):

            result = {}
            for index, label in index_label_pairs:
                if label in result:
                    result[label].add(index)
                else:
                    result[label] = set([index])
            return result

        def find_cue_label_by_image_idx(
            image_index,
            labels_to_indices_set,
            label_to_skip
        ):

            for label, indices_for_label \
                in labels_to_indices_set.items():

                if label == label_to_skip:
                    continue

                if image_index in indices_for_label:
                    return label

            assert False, \
                "Image index was not found " \
                "under any label for labels_to_indices_set"

        def equal_diag_off_diag_in_batch_warning(expression_equal_to_one):
            return "To use equal_diag_off_diag_in_batch == True " \
                    f"{expression_equal_to_one} " \
                    "== 1.0 is required"

        def duplicate_off_diag_samples(
            off_diag_triplet_indices,
            off_diag_multiplier,
            equal_diag_off_diag_in_batch,
            num_diag_samples
        ):
            assert off_diag_multiplier > 1

            off_diag_multiplier = np.ceil(off_diag_multiplier)

            off_diag_triplet_indices = np.repeat(
                np.array(off_diag_triplet_indices),
                int(off_diag_multiplier)
            ).tolist()

            if equal_diag_off_diag_in_batch:
                off_diag_triplet_indices \
                    = off_diag_triplet_indices[:num_diag_samples]

            return off_diag_triplet_indices

        def get_triplet_indices(
            diag_triplet_indices,
            off_diag_triplet_indices,
            shuffle,
            equal_diag_off_diag_in_batch
        ):

            triplet_indices = []

            if equal_diag_off_diag_in_batch:
                assert (
                    len(diag_triplet_indices) == len(off_diag_triplet_indices)
                )

                if shuffle:
                    random.shuffle(diag_triplet_indices)
                    random.shuffle(off_diag_triplet_indices)

                for diag_triplet_index, off_diag_triplet_index \
                    in zip(diag_triplet_indices, off_diag_triplet_indices):

                    triplet_indices.append(diag_triplet_index)
                    triplet_indices.append(off_diag_triplet_index)

            else:
                for diag_triplet_index in diag_triplet_indices:
                    triplet_indices.append(diag_triplet_index)
                for off_diag_triplet_index in off_diag_triplet_indices:
                    triplet_indices.append(off_diag_triplet_index)
                if shuffle:
                    random.shuffle(triplet_indices)

            return triplet_indices

        assert (
            off_diag_proportion >= 0
                and off_diag_proportion <= 1
        )

        if equal_diag_off_diag_in_batch:

            if off_diag_multiplier is not None:
                assert abs(
                    off_diag_proportion * off_diag_multiplier - 1.0
                ) < ERROR_EPS, \
                    equal_diag_off_diag_in_batch_warning(
                        "(off_diag_proportion * off_diag_multiplier)"
                    )

            else:
                assert off_diag_proportion == 1.0, \
                    equal_diag_off_diag_in_batch_warning(
                        "off_diag_proportion"
                    )

        assert_same_dsprites_holders(
            dataset_labelled_according_to_easy_to_bias_cue.dsprites_holder,
            dataset_labelled_according_to_ground_truth_cue.dsprites_holder
        )

        self.dsprites_holder \
            = dataset_labelled_according_to_easy_to_bias_cue.dsprites_holder

        labels_to_indices_for_easy_to_bias_cue = make_labels_to_indices_set(
            dataset_labelled_according_to_easy_to_bias_cue.index_label_pairs
        )

        labels_to_indices_for_ground_truth_cue = make_labels_to_indices_set(
            dataset_labelled_according_to_ground_truth_cue.index_label_pairs
        )

        assert (
            sorted(list(labels_to_indices_for_easy_to_bias_cue.keys()))
                == sorted(list(labels_to_indices_for_ground_truth_cue.keys()))
        )

        self.triplets = []

        diag_samples = {}
        for label in labels_to_indices_for_easy_to_bias_cue.keys():
            diag_samples[label] = set.intersection(
                labels_to_indices_for_easy_to_bias_cue[label],
                labels_to_indices_for_ground_truth_cue[label]
            )
            for idx in sorted(list(diag_samples[label])):
                self.triplets.append((idx, label, label))

        total_num_diag_samples = len(self.triplets)

        if off_diag_proportion > 0:
            for label in labels_to_indices_for_ground_truth_cue.keys():

                off_diag_for_label \
                    = labels_to_indices_for_ground_truth_cue[label].difference(
                        diag_samples[label]
                    )

                for image_index in sorted(list(off_diag_for_label)):
                    self.triplets.append((
                        image_index,
                        EMPTY_LABEL if unlabelled_off_diag else label,
                        find_cue_label_by_image_idx(
                            image_index,
                            labels_to_indices_for_easy_to_bias_cue,
                            label
                        )
                    ))

        if dataset_size == 0:
            dataset_size = len(self.triplets)

        diag_triplet_indices = list(range(total_num_diag_samples))
        off_diag_triplet_indices = list(
            range(total_num_diag_samples, len(self.triplets))
        )

        if equal_diag_off_diag_in_batch and off_diag_multiplier is None:
            assert dataset_size % 2 == 0, \
                "Should have even dataset size " \
                "when equal_diag_off_diag_in_batch == True " \
                "and off_diag_multiplier is None"

        requested_num_diag_samples \
            = int(dataset_size / (1 + off_diag_proportion))

        assert requested_num_diag_samples > 0

        assert requested_num_diag_samples <= total_num_diag_samples, \
            "Can not have as many diag samples as needed " \
            "for off-diag proportion to hold. " \
            "Requested: {}, but have in total: {}".format(
                requested_num_diag_samples,
                total_num_diag_samples
            )

        requested_num_off_diag_samples \
            = dataset_size - requested_num_diag_samples

        if off_diag_proportion > 0:
            assert requested_num_off_diag_samples > 0

        proportion_error = abs(
            float(requested_num_off_diag_samples / requested_num_diag_samples)
                - off_diag_proportion
        )
        assert proportion_error < ERROR_EPS

        diag_triplet_indices = utils.subsample_list_uniformly(
            diag_triplet_indices,
            requested_num_diag_samples
        )
        off_diag_triplet_indices = utils.subsample_list_uniformly(
            off_diag_triplet_indices,
            requested_num_off_diag_samples
        )

        if off_diag_multiplier is not None:

            off_diag_triplet_indices = duplicate_off_diag_samples(
                off_diag_triplet_indices,
                off_diag_multiplier,
                equal_diag_off_diag_in_batch,
                len(diag_triplet_indices)
            )

        self.triplet_indices = get_triplet_indices(
            diag_triplet_indices,
            off_diag_triplet_indices,
            shuffle,
            equal_diag_off_diag_in_batch
        )

    def __getitem__(self, idx):
        assert len(self.triplets) > 0
        assert len(self.triplet_indices) > 0
        index_in_dsprites_holder, ground_truth_label, easy_to_bias_cue_label \
            = self.triplets[self.triplet_indices[idx]]
        return (
            torch.Tensor(self.dsprites_holder[index_in_dsprites_holder]),
            torch.tensor(ground_truth_label, dtype=torch.int64),
            torch.tensor(easy_to_bias_cue_label, dtype=torch.int64)
        )

    def __len__(self):
        return len(self.triplet_indices)


def make_dataset_with_diag_and_off_diag_cells_from_config(
    dataset_with_diag_and_off_diag_cells_config
):

    cache_path = dataset_with_diag_and_off_diag_cells_config["cache_path"]
    equal_diag_off_diag_in_batch = dataset_with_diag_and_off_diag_cells_config[
        "equal_diag_off_diag_in_batch"
    ]
    dataset_config = {
        "dsprites_holder_args": dataset_with_diag_and_off_diag_cells_config[
            "dsprites_holder_args"
        ],
        "num_classes": dataset_with_diag_and_off_diag_cells_config[
            "num_classes"
        ],
        "cue": dataset_with_diag_and_off_diag_cells_config[
            "easy_to_bias_cue"
        ],
        "split": dataset_with_diag_and_off_diag_cells_config["split"]
    }

    save_func, load_func = get_save_load_funcs(
        ["dsprites_holder"],
        dataset_config["dsprites_holder_args"]
    )

    easy_to_bias_cue_dataset = utils.make_or_load_from_cache(
        "one_cue_dataset",
        dataset_config,
        make_dataset_labelled_according_to_cue,
        cache_path=cache_path,
        save_func=save_func,
        load_func=load_func
    )

    dataset_config["cue"] = dataset_with_diag_and_off_diag_cells_config[
        "ground_truth_cue"
    ]

    ground_truth_cue_dataset = utils.make_or_load_from_cache(
        "one_cue_dataset",
        dataset_config,
        make_dataset_labelled_according_to_cue,
        cache_path=cache_path,
        save_func=save_func,
        load_func=load_func
    )

    return DatasetWithDiagAndOffDiagCells(
        easy_to_bias_cue_dataset,
        ground_truth_cue_dataset,
        off_diag_proportion=dataset_with_diag_and_off_diag_cells_config[
            "off_diag_proportion"
        ],
        unlabelled_off_diag\
            =dataset_with_diag_and_off_diag_cells_config["unlabelled_off_diag"],
        equal_diag_off_diag_in_batch=equal_diag_off_diag_in_batch,
        shuffle=(
            dataset_with_diag_and_off_diag_cells_config["shuffle"]
                if equal_diag_off_diag_in_batch
                else False
        ),
        dataset_size\
            =dataset_with_diag_and_off_diag_cells_config["dataset_size"],
        off_diag_multiplier\
            =dataset_with_diag_and_off_diag_cells_config["off_diag_multiplier"]
    )


def get_dataloaders_for_all_cues(
    dsprites_holder,
    num_classes,
    batch_size,
    multilabel=False,
    split="train",
    shuffle=True,
    dataset_size=TRAIN_DATASET_SIZE
):

    def get_dataloader(
        dsprites_holder,
        cue,
        num_classes,
        batch_size,
        split,
        shuffle,
        dataset_size
    ):
        dataset = DatasetLabelledAccordingToCue(
            dsprites_holder=dsprites_holder,
            num_classes=num_classes,
            cue=cue,
            split=split
        )
        subsampled_indices = utils.deterministically_subsample_indices_uniformly(
            len(dataset),
            dataset_size
        )
        dataset = torch.utils.data.Subset(dataset, subsampled_indices)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def get_multilabel_dataloader(
        dsprites_holder,
        num_classes,
        batch_size,
        split,
        shuffle,
        dataset_size
    ):

        multilabel_dataset = MultilabelDataset(
            dsprites_holder,
            num_classes,
            split
        )

        subsampled_indices \
            = utils.deterministically_subsample_indices_uniformly(
                len(multilabel_dataset),
                dataset_size
            )

        subsampled_multilabel_dataset = torch.utils.data.Subset(
            multilabel_dataset,
            subsampled_indices
        )
        dataloader = torch.utils.data.DataLoader(
            subsampled_multilabel_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        dataloader.cue_names = multilabel_dataset.cue_names

        return dataloader

    if multilabel:

        multilabel_dataloader = get_multilabel_dataloader(
            dsprites_holder,
            num_classes,
            batch_size=batch_size,
            split=split,
            shuffle=shuffle,
            dataset_size=dataset_size
        )

        return {"all_cues": multilabel_dataloader}

    else:

        dataloaders = {}

        for cue in dsprites_holder.latents_names:

            if (not dsprites_holder.colored) and cue == "color":
                continue

            dataloaders[cue] = get_dataloader(
                dsprites_holder,
                cue,
                num_classes,
                batch_size=batch_size,
                split=split,
                shuffle=shuffle,
                dataset_size=dataset_size
            )

        return dataloaders


def make_dataloaders_for_all_cues_from_config(dataloaders_config):

    return get_dataloaders_for_all_cues(
        dsprites_holder=prepare_dsprites_holder_maker(
            **dataloaders_config["dsprites_holder_args"]
        )(),
        num_classes=dataloaders_config["num_classes"],
        batch_size=dataloaders_config["batch_size"],
        multilabel=dataloaders_config["multilabel"],
        split=dataloaders_config["split"],
        shuffle=dataloaders_config["shuffle"],
        dataset_size=dataloaders_config["dataset_size"]
    )


def get_diag_and_off_diag_dataloader(
    dsprites_holder_args,
    num_classes,
    easy_to_bias_cue,
    ground_truth_cue,
    off_diag_proportion,
    split,
    dataset_size,
    shuffle,
    batch_size,
    cache_path,
    off_diag_multiplier=None,
    unlabelled_off_diag=False,
    equal_diag_off_diag_in_batch=False
):

    dataset_with_diag_and_off_diag_cells_config = {
        "dsprites_holder_args": dsprites_holder_args,
        "num_classes": num_classes,
        "easy_to_bias_cue": easy_to_bias_cue,
        "ground_truth_cue": ground_truth_cue,
        "off_diag_proportion": off_diag_proportion,
        "cache_path": cache_path,
        "off_diag_multiplier": off_diag_multiplier,
        "unlabelled_off_diag": unlabelled_off_diag,
        "equal_diag_off_diag_in_batch": equal_diag_off_diag_in_batch,
        "split": split,
        "dataset_size": dataset_size,
        "shuffle": shuffle
    }

    save_func, load_func = get_save_load_funcs(
        ["dsprites_holder"],
        dataset_with_diag_and_off_diag_cells_config["dsprites_holder_args"]
    )

    diag_and_off_diag_dataset = utils.make_or_load_from_cache(
        "diag_off_diag_dataset",
        dataset_with_diag_and_off_diag_cells_config,
        make_dataset_with_diag_and_off_diag_cells_from_config,
        cache_path=cache_path,
        save_func=save_func,
        load_func=load_func
    )

    return torch.utils.data.DataLoader(
        diag_and_off_diag_dataset,
        batch_size=batch_size,
        shuffle=(False if equal_diag_off_diag_in_batch else shuffle)
    )


def prepare_dsprites_holder_maker(
    train_val_split,
    colored,
    cache_path
):

    dsprites_holder_config = {
        "train_val_split": train_val_split,
        "colored": colored
    }

    def make_dsprites_holder():
        return utils.make_or_load_from_cache(
            "dsprites_holder",
            dsprites_holder_config,
            make_dsprites_holder_from_config,
            cache_path=cache_path
        )

    return make_dsprites_holder


def prepare_default_dsprites_dataloaders_maker(
    dsprites_holder_args: Dict[str, Any] \
        = DEFAULT_DSPRITES_HOLDER_ARGS,
    num_classes: int = NUM_CLASSES,
    batch_size: int = BATCH_SIZE,
    is_multilabel: bool = False,
    cache_path: str = CACHE_PATH,
    split: str = "train",
    shuffle: bool = True,
    dataset_size: int = TRAIN_DATASET_SIZE,
    one_dataloader_to_select: str = None
) -> Callable:
    """
    Prepare a factory function for creating default dsprites dataloaders
    by optionally caching them.

    Where default dsprites dataloader is the one
    that generates tuples of input image from the dSprites dataset
    and it's labels.

    If <is_multilabel> is True, then (each) dataloader
    will have an attribute "cue_names" which will store
    the correctly ordered names of cues for which the labels are given.

    Args:

        dsprites_holder_args (dict[str, Any]): keyword args
            for local class method "DSpritesHolder.__init__()"
            and cache_path keyword arg that does the same as <cache_path>
            for the current function.
            Default: DEFAULT_DSPRITES_HOLDER_ARGS

        num_classes (int): number of classes into which to distribute
            all possible values of each cue.

            For example if cue "scale" has 6 values
            {scale_0, scale_1, scale_2, scale_3, scale_4, scale_5},
            then if <num_classes> equals to 3,
            image classes will be assigned in the following way:
                class 0: (scale == scale_0 or scale == scale_1)
                class 1: (scale == scale_2 or scale == scale_3)
                class 2: (scale == scale_4 or scale == scale_5).

            When the number of cue values (n_values) is not divisible
            by the <n_classes>, each class gets (n_values / <n_classes>) values
            and first (n_values % <n_classes>) classes get one additional value.
            Where / means an integer division and % means a remainder
            from the integer division.
            By "class gets k values" we mean that to assign an image
            to this class, its cue value needs to be equal to one of k
            different corresponding cue values distributed to this class.
            Default: NUM_CLASSES

        batch_size (int): batch size for (each) created dataloader.
            Default: BATCH_SIZE

        is_multilabel (bool): a flag, if it is True then one dataloader
            is created for all cues.
            It will generate an input image batch and a list of labels batches
            for each cue in the order specified
            by this dataloader's "cue_names" attribute.
            If it is False, then a separate dataloader is created for each cue.
            Each of these dataloaders will generate an input image batch
            and one labels batch according to the corresponding cue.
            Default: False

        cache_path (str): arg for local function "utils.make_or_load_from_cache"
            used for making the dataloader.
            Default: CACHE_PATH

        split (str): a split name for samples
            (splits are defined in local class "DSpritesHolder").
            Default: "train"

        shuffle (bool): a flag, if it is True samples will be shuffled.
            Default: True

        dataset_size (int): number of samples used by (each) created dataloader.
            Default: TRAIN_DATASET_SIZE

        one_dataloader_to_select (str): a name of the only dataloader
            to return (possible names are cue names if <is_multilabel> is False
            or "all_cues" otherwise).
            If it is None, then this argument is ignored.
            Default: None

    Returns:

        a factory function that takes model as an argument
        (needed for compatibility with adversarial dataloaders)
        and creates dataloaders in one of the following formats:

            - torch.utils.data.DataLoader: the only one returned dataloader
                if <one_dataloader_to_select> is not None.

            - Dict[str, torch.utils.data.DataLoader]: a dictionary
                that maps cue name to the dataloader with corresponding labels
                if <one_dataloader_to_select> is None.

                If <is_multilabel> is True
                and <one_dataloader_to_select> is None
                then the only key in the returned dictionary will be "all_cues".
    """

    dataloaders_config = {
        "dsprites_holder_args": dsprites_holder_args,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "multilabel": is_multilabel,
        "split": split,
        "shuffle": shuffle,
        "dataset_size": dataset_size
    }

    def make_default_dsprites_dataloaders(model) \
        -> Union[
            torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]
        ]:

        save_func, load_func = get_save_load_funcs(
            ["dataset", "dataset", "dsprites_holder"],
            dataloaders_config["dsprites_holder_args"]
        )

        dataloaders = utils.make_or_load_from_cache(
            "default_dsprites_dataloaders",
            dataloaders_config,
            make_dataloaders_for_all_cues_from_config,
            cache_path=cache_path,
            save_func=save_func,
            load_func=load_func
        )

        if one_dataloader_to_select is not None:
            assert one_dataloader_to_select in dataloaders
            return dataloaders[one_dataloader_to_select]

        return dataloaders

    return make_default_dsprites_dataloaders


def prepare_de_biasing_task_dataloader_maker(
    dsprites_holder_args: Dict[str, Any] \
        = DEFAULT_DSPRITES_HOLDER_ARGS,
    num_classes: int = NUM_CLASSES,
    batch_size: int = BATCH_SIZE,
    easy_to_bias_cue: str = EASY_TO_BIAS_CUE,
    ground_truth_cue: str = GROUND_TRUTH_CUE,
    off_diag_proportion: float = 0,
    off_diag_multiplier: int = None,
    unlabelled_off_diag: bool = False,
    equal_diag_off_diag_in_batch: bool = False,
    cache_path: str = CACHE_PATH,
    split: str = "train",
    shuffle: bool = True,
    dataset_size: int = TRAIN_DATASET_SIZE,
) -> Callable:
    """
    Prepare a factory function for creating dataloaders for the de-biasing task
    by optionally caching them.

    It will create two types of dataloaders: "diag_and_off_diag_dataloader"
    and "default_dataloader":

        - "default_dataloader" is the one created by the factory function
        prepared in local function "prepare_default_dsprites_dataloaders_maker".

        - "diag_and_off_diag_dataloader" is the one that generates tuples of:
            - images
            - "ground truth" labels
            - "easy to bias" labels.

        Where the "ground truth" labels are the correct labels
        that are assigned according to the cue
        which model is trained to distinguish;
        "easy to bias" labels are the labels that are assigned to the cue
        which model easily gets biased to.

        Samples for which "ground truth" labels coincide
        with "easy to bias" labels are called "diagonal" samples,
        otherwise samples are called "off-diagonal".

    Args:

        dsprites_holder_args, num_classes, batch_size, cache_path,
        split, shuffle, dataset_size:
            same as in local function
            "prepare_default_dsprites_dataloaders_maker".
            Defaults are also the same as in that function.

        easy_to_bias_cue (str): name of the cue
            according to which "easy to bias" labels are assigned.
            Default: EASY_TO_BIAS_CUE

        ground_truth_cue (str): name of the cue
            according to which "ground truth" labels are assigned.
            Default: GROUND_TRUTH_CUE

        off_diag_proportion (float): a ratio: the number of "off-diagonal"
            divided by the number of "diagonal" samples.
            Default: 0

        off_diag_multiplier (int): a number of duplicates
            each "off-diagonal" sample will have to artificially increase
            their number in each batch. If None, no duplicates are added.
            Default: None

        unlabelled_off_diag (bool): a flag, if it is True,
            "off-diagonal" samples have ground truth labels,
            otherwise their labels equal to the EMPTY_LABEL.
            Default: True

        equal_diag_off_diag_in_batch (bool): a flag if it is True,
            the number of "diagonal" samples in a batch is the same as
            the number of "off-diagonal" samples in a batch,
            otherwise, these numbers might not be the same.

            If it is True, then the following should hold:
            (
                <off_diag_proportion> == 1.0
                    or (<off_diag_proportion> * <off_diag_multiplier>) == 1.0
            ).
            Default: False

    Returns:

        a factory function that takes model as an argument
        (needed for compatibility with adversarial dataloaders)
        and creates dataloaders in one of the following formats:

            - "diag_and_off_diag_dataloader", if <split> is "train".

            - Dict[
                str, Union["diag_and_off_diag_dataloader", "default_dataloader"]
            ] with the following contents:

                - "test_name": "diag_and_off_diag_dataloader"

                - <ground_truth_cue>: "default_dataloader",

                if <split> is "test".

                Where "test_name" is
                "diag+(off_diag_proportion=<off_diag_proportion>)"
                if <off_diag_proportion> > 0 and just "diag" otherwise.

            Where "diag_and_off_diag_dataloader" and "default_dataloader"
            are both of torch.utils.data.DataLoader type.

            "default_dataloader" is created
            using "prepare_default_dsprites_dataloaders_maker"
            by forwarding arguments from this function
            and setting unique additional argument <one_dataloader_to_select>
            of the former equal to the <ground_truth_cue>.
    """

    def make_debiasing_task_dataloader(model) \
        -> Union[
            torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]
        ]:

        diag_dataloader = get_diag_and_off_diag_dataloader(
            dsprites_holder_args=dsprites_holder_args,
            num_classes=num_classes,
            easy_to_bias_cue=easy_to_bias_cue,
            ground_truth_cue=ground_truth_cue,
            off_diag_proportion=off_diag_proportion,
            split=split,
            dataset_size=dataset_size,
            shuffle=shuffle,
            batch_size=batch_size,
            cache_path=cache_path,
            off_diag_multiplier=off_diag_multiplier,
            unlabelled_off_diag=unlabelled_off_diag,
            equal_diag_off_diag_in_batch=equal_diag_off_diag_in_batch
        )

        if split == "test":
            ground_truth_cue_test_dataloader \
                = prepare_default_dsprites_dataloaders_maker(
                    dsprites_holder_args=dsprites_holder_args,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    cache_path=cache_path,
                    split=split,
                    shuffle=shuffle,
                    dataset_size=dataset_size,
                    one_dataloader_to_select=ground_truth_cue
                )(None)
            test_name = "diag"
            if off_diag_proportion > 0:
                test_name += f"+(off_diag_proportion={off_diag_proportion})"
            diag_dataloader = {test_name: diag_dataloader}
            diag_dataloader[ground_truth_cue] = ground_truth_cue_test_dataloader

        return diag_dataloader

    return make_debiasing_task_dataloader


def set_attr_for_data_object(data_object, nested_data_holder_attr, value):
    if utils.has_nested_attr(data_object, nested_data_holder_attr):
        utils.set_nested_attr(data_object, nested_data_holder_attr, value)
    else:
        assert isinstance(data_object, dict)
        assert len(data_object) > 0
        for data_subobject in data_object.values():
            assert utils.has_nested_attr(
                data_subobject,
                nested_data_holder_attr
            )
            utils.set_nested_attr(
                data_subobject,
                nested_data_holder_attr,
                value
            )


def get_attr_from_data_object(data_object, nested_data_holder_attr):
    if utils.has_nested_attr(data_object, nested_data_holder_attr):
        return utils.get_nested_attr(data_object, nested_data_holder_attr)
    else:
        res = None
        assert isinstance(data_object, dict)
        assert len(data_object) > 0
        for data_subobject in data_object.values():
            assert utils.has_nested_attr(
                data_subobject,
                nested_data_holder_attr
            )
            if res is None:
                res = utils.get_nested_attr(
                    data_subobject,
                    nested_data_holder_attr
                )
            else:
                assert (
                    res is utils.get_nested_attr(
                        data_subobject,
                        nested_data_holder_attr
                    )
                )
        assert res is not None
        return res


def prepare_object_without_data_holder_saver(nested_data_holder_attr):

    def save_data(data_object, path):
        data_holder = get_attr_from_data_object(
            data_object,
            nested_data_holder_attr
        )
        set_attr_for_data_object(data_object, nested_data_holder_attr, None)
        utils.default_save_func(data_object, path)
        set_attr_for_data_object(
            data_object,
            nested_data_holder_attr,
            data_holder
        )

    return save_data


def prepare_object_without_data_holder_loader(
    nested_data_holder_attr,
    make_data_holder
):

	data_holder = make_data_holder()

	def load_data(path):
		data_object = utils.default_load_func(path)
		set_attr_for_data_object(
            data_object,
            nested_data_holder_attr,
            data_holder
        )
		return data_object

	return load_data


def get_save_load_funcs(nested_dsprites_holder_attr, dsprites_holder_args):

    save_func = prepare_object_without_data_holder_saver(
        nested_dsprites_holder_attr
    )
    load_func = prepare_object_without_data_holder_loader(
        nested_dsprites_holder_attr,
        prepare_dsprites_holder_maker(
            **dsprites_holder_args
        )
    )
    return save_func, load_func

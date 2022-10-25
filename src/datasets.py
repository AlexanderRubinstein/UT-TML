import numpy as np
import torch
import random


# local modules
import utils


DSPRITES_NPZ_PATH = "/tmp/dsprites.npz"
DSPRITES_NUM_COLORS = 3
EMPTY_LABEL = -1
EPS = 1e-3
# BATCH_SIZE = 64
NUM_CLASSES = 3
TRAIN_DATASET_SIZE = 20000
TEST_DATASET_SIZE = 2000
CACHE_PATH = "/tmp/ood_cache"


class DSpritesHolder:
    def __init__(self, train_val_split, colored=False):
        dsprites_zip = read_dsprites_npz(DSPRITES_NPZ_PATH)
        self.imgs = dsprites_zip['imgs']
        metadata = dsprites_zip['metadata'][()]
        self.latents_names = metadata['latents_names']
        self.latents_names_to_indices = {}
        for i, name in enumerate(self.latents_names):
            self.latents_names_to_indices[name] = i
        self.latents_possible_values = metadata['latents_possible_values']
        self.latents_sizes = metadata['latents_sizes']
        self.latents_bases = np.concatenate(
            (self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,]))
        )

        self.train_val_split = train_val_split

        self.num_images_of_one_color = self.imgs.shape[0]
        self.colored = colored
        self.latents_sizes[0] = DSPRITES_NUM_COLORS if self.colored else 1
        self.colored_image_shape \
            = (self.latents_sizes[0],) \
                + self.imgs[0].shape
        self.total_num_images \
            = self.num_images_of_one_color * self.latents_sizes[0]

        self.indices = {}
        self.indices["train"], self.indices["test"] \
            = self._split_into_train_and_test()


    def _latents_values_combination_to_index(
        self,
        latents_values_combination
    ):
        assert isinstance(latents_values_combination, np.ndarray)
        return np.dot(
            latents_values_combination,
            self.latents_bases
        ).astype(int)

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
            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            assert self.colored_image_shape
            color = int(idx / self.num_images_of_one_color)
            original_idx = idx % self.num_images_of_one_color
            colored_image = np.zeros(self.colored_image_shape)
            colored_image[color, ...] = self.imgs[original_idx]
            image = colored_image
            ######### ANSWER END
        else:
            image = self.imgs[idx][None, ...]

        return image

    ######### ANSWER BEGIN
#   TODO(Alex | 13.09.2022) Use slicing of np.array instead of set stuff
    def _split_into_train_and_test(self):
        assert self.total_num_images
        train_indices = []
        test_indices = []
        all_indices = list(range(self.total_num_images))

        train_positions = set(
            np.linspace(
                0,
                len(all_indices) - 1,
                utils.compute_proportion(self.train_val_split, len(all_indices)),
                dtype=int
            )
        )

        for i, idx in enumerate(all_indices):
            if i in train_positions:
                train_indices.append(idx)
            else:
                test_indices.append(idx)
        return train_indices, test_indices
    ######### ANSWER END


def make_dsprites_holder(dsprites_holder_config):
    return DSpritesHolder(
        dsprites_holder_config["train_val_split"],
        colored=dsprites_holder_config["colored"]
    )


def read_dsprites_npz(filename):
    return np.load(filename, allow_pickle=True, encoding='latin1')


class DatasetLabelledAccordingToCue:

    def __init__(self, dsprites_holder, num_classes, cue, split):

        def distribute_latent_values_to_classes(num_classes, latent_size):
            assert num_classes <= latent_size
            # ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            # ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            num_per_class = int(latent_size / num_classes)
            remainder = latent_size % num_classes

            max_latent_values_per_class = [
                (
                    (class_id + 1)
                        * num_per_class
                        + min(class_id, remainder - 1)
                        + 1
                )
                    for class_id in range(num_classes)
            ]
            ######### ANSWER END

            assert max_latent_values_per_class[-1] == latent_size
            assert isinstance(max_latent_values_per_class, list)
            assert len(max_latent_values_per_class) == num_classes

            return max_latent_values_per_class

        def dict_latent_value_id_to_label(max_latent_values_per_class):
            # ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            # ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            latent_value_id_to_label = {}

            latent_size = max_latent_values_per_class[-1]
            class_id = 0
            for latent_value in range(latent_size):
                if latent_value >= max_latent_values_per_class[class_id]:
                    class_id += 1
                latent_value_id_to_label[latent_value] = class_id
            ######### ANSWER END

            assert isinstance(latent_value_id_to_label, dict)
            assert (
                len(latent_value_id_to_label)
                    == num_latent_values_per_class[-1]
            )

            return latent_value_id_to_label

        self.dsprites_holder = dsprites_holder
        cue_id = self.dsprites_holder.latents_names_to_indices[cue]
        num_latent_values_per_class = distribute_latent_values_to_classes(
            num_classes,
            self.dsprites_holder.latents_sizes[cue_id]
        )
        self.latent_values_to_labels = dict_latent_value_id_to_label(
            num_latent_values_per_class
        )
        self.index_label_pairs = []

        for index in self.dsprites_holder.indices[split]:
            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            latent_values_combination \
                = self.dsprites_holder._index_to_values_combination(index)
            latent_value_id = latent_values_combination[cue_id]
            label = self.latent_values_to_labels[latent_value_id]
            self.index_label_pairs.append((index, label))
            ######### ANSWER END

    def __getitem__(self, idx_of_item_from_this_dataset):
        assert len(self.index_label_pairs) > 0
        index, label = self.index_label_pairs[idx_of_item_from_this_dataset]

        return (torch.Tensor(
            self.dsprites_holder[index]),
            torch.tensor(label, dtype=torch.int64)
        )

    def __len__(self):
        return len(self.index_label_pairs)


def make_dataset_labelled_according_to_cue(dataset_config):
    return DatasetLabelledAccordingToCue(
        dataset_config["dsprites_holder"],
        dataset_config["num_classes"],
        dataset_config["cue"],
        dataset_config["split"]
    )


class MultilabelDataset:
    def __init__(self, dsprites_holder, num_classes, split):
        image_to_labels = {}
        self.dsprites_holder = dsprites_holder
        self.cue_names = []
        for cue in self.dsprites_holder.latents_names:
            self.cue_names.append(cue)

            one_cue_dataset = DatasetLabelledAccordingToCue(
                self.dsprites_holder,
                num_classes,
                cue,
                split
            )

            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            for image_idx, label in one_cue_dataset.index_label_pairs:
                # if image_idx not in image_to_labels:
                #     image_to_labels[image_idx] = [label]
                # else:
                #     image_to_labels[image_idx].append(label)
                utils.append_to_list_in_dict(image_to_labels, image_idx, label)
            ######### ANSWER END

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
        off_diag_samples_proportion,
        unlabelled_off_diag,
        equal_diag_off_diag_in_batch,
        shuffle,
        dataset_size,
        off_diag_multiplier
    ):

        def make_indices_to_labels_set(index_label_pairs):
            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            result = {}
            for index, label in index_label_pairs:
                if label in result:
                    result[label].add(index)
                else:
                    result[label] = set([index])
            return result
            ######### ANSWER END

        def find_easy_to_bias_cue_label(
            image_index,
            indices_to_labels_for_easy_to_bias_cue
        ):
            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            for label, indices_for_label \
                in indices_to_labels_for_easy_to_bias_cue.items():

                if image_index in indices_for_label:
                    return label
            assert False, "Image index not found under any label for strong cue"
            ######### ANSWER END

        if equal_diag_off_diag_in_batch:
            if off_diag_multiplier is not None:
                assert abs(
                    off_diag_samples_proportion * off_diag_multiplier - 1.0
                ) < EPS, \
                    "To use equal_diag_off_diag_in_batch=True " \
                    "(off_diag_samples_proportion * off_diag_multiplier) " \
                    "== 1.0 is required"
            else:
                assert off_diag_samples_proportion == 1.0, \
                    "To use equal_diag_off_diag_in_batch=True " \
                    "off_diag_samples_proportion == 1.0 " \
                    "is required"

        self.dsprites_holder \
            = dataset_labelled_according_to_easy_to_bias_cue.dsprites_holder

        indices_to_labels_for_easy_to_bias_cue = make_indices_to_labels_set(
            dataset_labelled_according_to_easy_to_bias_cue.index_label_pairs
        )
        indices_to_labels_for_ground_truth_cue = make_indices_to_labels_set(
            dataset_labelled_according_to_ground_truth_cue.index_label_pairs
        )

        assert (
            sorted(list(indices_to_labels_for_easy_to_bias_cue.keys()))
                == sorted(list(indices_to_labels_for_ground_truth_cue.keys()))
        )

        self.triplets = []

        diag_samples = {}
        for label in indices_to_labels_for_easy_to_bias_cue.keys():
            diag_samples[label] = set.intersection(
                indices_to_labels_for_easy_to_bias_cue[label],
                indices_to_labels_for_ground_truth_cue[label]
            )
            for idx in diag_samples[label]:
                self.triplets.append((idx, label, label))

        total_num_diag_samples = len(self.triplets)

        if off_diag_samples_proportion > 0:
            for label in indices_to_labels_for_easy_to_bias_cue.keys():
                diag_samples_for_label = diag_samples[label]
                off_diag_for_label \
                    = indices_to_labels_for_ground_truth_cue[label].difference(
                        diag_samples_for_label
                    )

                for i, image_index in enumerate(off_diag_for_label):
                    self.triplets.append((
                        image_index,
                        EMPTY_LABEL if unlabelled_off_diag else label,
                        find_easy_to_bias_cue_label(
                            image_index,
                            indices_to_labels_for_easy_to_bias_cue
                        )
                    ))

        if dataset_size == 0:
            dataset_size = len(self.triplets)

        diag_triplet_indices = list(range(total_num_diag_samples))
        off_diag_triplet_indices = list(
            range(total_num_diag_samples,
            len(self.triplets))
        )

        if equal_diag_off_diag_in_batch:
            assert dataset_size % 2 == 0, \
                "Should have even size of dataset" \
                " when equal_diag_off_diag_in_batch==True"

        num_diag_samples = int(dataset_size / (1 + off_diag_samples_proportion))

        assert num_diag_samples <= total_num_diag_samples, \
            "Can not have as many diag samples as needed " \
            "for off-diag proportion to hold"

        num_off_diag_samples = dataset_size - num_diag_samples
        proportion_error = abs(
            float(num_off_diag_samples / num_diag_samples)
                - off_diag_samples_proportion
        )
        assert proportion_error < EPS

        diag_triplet_indices = utils.subsample_list_by_indices(
            diag_triplet_indices,
            num_diag_samples
        )
        off_diag_triplet_indices = utils.subsample_list_by_indices(
            off_diag_triplet_indices,
            num_off_diag_samples
        )

        if off_diag_multiplier is not None:

            assert off_diag_multiplier > 1

            off_diag_multiplier = np.ceil(off_diag_multiplier)

            off_diag_triplet_indices = np.repeat(
                np.array(off_diag_triplet_indices),
                int(off_diag_multiplier)
            ).tolist()

            if equal_diag_off_diag_in_batch:
                off_diag_triplet_indices \
                    = off_diag_triplet_indices[:len(diag_triplet_indices)]

        self.triplet_indices = []

        if equal_diag_off_diag_in_batch:
            assert len(diag_triplet_indices) == len(off_diag_triplet_indices)
            if shuffle:
                random.shuffle(diag_triplet_indices)
                random.shuffle(off_diag_triplet_indices)
            ######### ATTENTION PLEASE
            pass  # please put your code instead of this line
            ######### THANK YOU FOR YOUR ATTENTION

            ######### ANSWER BEGIN
            for diag_triplet_index, off_diag_triplet_index \
                in zip(diag_triplet_indices, off_diag_triplet_indices):

                self.triplet_indices.append(diag_triplet_index)
                self.triplet_indices.append(off_diag_triplet_index)
            ######### ANSWER END
        else:
            for diag_triplet_index in diag_triplet_indices:
                self.triplet_indices.append(diag_triplet_index)
            for off_diag_triplet_index in off_diag_triplet_indices:
                self.triplet_indices.append(off_diag_triplet_index)
            if shuffle:
                random.shuffle(self.triplet_indices)

    def __getitem__(self, idx):
        assert len(self.triplets) > 0
        index, label, image_type = self.triplets[self.triplet_indices[idx]]
        return (torch.Tensor(
            self.dsprites_holder[index]),
            torch.tensor(label, dtype=torch.int64),
            image_type
        )

    def __len__(self):
        return len(self.triplet_indices)


# def get_dataloaders_for_all_cues(
#     dsprites_holder,
#     num_classes,
#     batch_size,
#     # num_classes=NUM_CLASSES,
#     # batch_size=BATCH_SIZE,
#     multilabel=False
# ):
def get_dataloaders_for_all_cues(
    dsprites_holder,
    num_classes,
    batch_size,
    # num_classes=NUM_CLASSES,
    # batch_size=BATCH_SIZE,
    multilabel=False,
    split="train",
    shuffle=True,
    dataset_size=TRAIN_DATASET_SIZE
):

    def get_dataloader(
        dsprites_holder,
        num_classes,
        cue,
        split,
        dataset_size,
        shuffle,
        batch_size
    ):
        dataset = DatasetLabelledAccordingToCue(
            dsprites_holder,
            num_classes,
            cue,
            split
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

    def get_multilabel_dataset(
        dsprites_holder,
        num_classes,
        split,
        dataset_size,
        shuffle,
        batch_size
    ):

        multilabel_dataset = MultilabelDataset(
            dsprites_holder,
            num_classes,
            split
        )
        subsampled_indices = utils.deterministically_subsample_indices_uniformly(
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
        # multilabel_train_dataloader = get_multilabel_dataset(
        #     dsprites_holder,
        #     num_classes,
        #     split="train",
        #     dataset_size=TRAIN_DATASET_SIZE,
        #     shuffle=True,
        #     batch_size=batch_size
        # )
        # multilabel_test_dataloader = get_multilabel_dataset(
        #     dsprites_holder,
        #     num_classes,
        #     split="test",
        #     dataset_size=TEST_DATASET_SIZE,
        #     shuffle=False,
        #     batch_size=batch_size
        # )
        multilabel_dataloader = get_multilabel_dataset(
            dsprites_holder,
            num_classes,
            split=split,
            dataset_size=dataset_size,
            shuffle=shuffle,
            batch_size=batch_size
        )
        return {"all_cues": multilabel_dataloader}

    else:
        dataloaders = {}
        # test_dataloaders = {}
        for cue in dsprites_holder.latents_names:
            if (not dsprites_holder.colored) and cue == "color":
                continue

            # train_dataloaders[cue] = get_dataloader(
            #     dsprites_holder,
            #     num_classes,
            #     cue,
            #     split="train",
            #     dataset_size=TRAIN_DATASET_SIZE,
            #     shuffle=True,
            #     batch_size=batch_size
            # )
            # test_dataloaders[cue] = get_dataloader(
            #     dsprites_holder,
            #     num_classes,
            #     cue,
            #     split="test",
            #     dataset_size=TEST_DATASET_SIZE,
            #     shuffle=False,
            #     batch_size=batch_size
            # )
            dataloaders[cue] = get_dataloader(
                dsprites_holder,
                num_classes,
                cue,
                split=split,
                dataset_size=dataset_size,
                shuffle=shuffle,
                batch_size=batch_size
            )
        # return train_dataloaders, test_dataloaders
        return dataloaders

def make_dataloaders_for_all_cues_from_config(dataloaders_config):
    # return get_dataloaders_for_all_cues(
    #     dataloaders_config["dsprites_holder"],
    #     num_classes=dataloaders_config["num_classes"],
    #     batch_size=dataloaders_config["batch_size"],
    #     multilabel=dataloaders_config["multilabel"]
    # )
    return get_dataloaders_for_all_cues(
        dataloaders_config["dsprites_holder"],
        num_classes=dataloaders_config["num_classes"],
        batch_size=dataloaders_config["batch_size"],
        multilabel=dataloaders_config["multilabel"],
        split=dataloaders_config["split"],
        shuffle=dataloaders_config["shuffle"],
        dataset_size=dataloaders_config["dataset_size"]
    )


def get_diag_and_off_diag_dataloader(
    dsprites_holder,
    num_classes,
    easy_to_bias_cue,
    ground_truth_cue,
    off_diag_samples_proportion,
    split,
    dataset_size,
    shuffle,
    batch_size,
    off_diag_multiplier=None,
    unlabelled_off_diag=False,
    equal_diag_off_diag_in_batch=False
):

    dataset_config = {
        "dsprites_holder": dsprites_holder,
        "num_classes": num_classes,
        "cue": None,
        "split": split
    }

    dataset_config["cue"] = easy_to_bias_cue

    easy_to_bias_cue_dataset = utils.make_or_load_from_cache(
        "one_cue_dataset",
        dataset_config,
        make_dataset_labelled_according_to_cue,
        cache_path=CACHE_PATH
    )

    dataset_config["cue"] = ground_truth_cue

    ground_truth_cue_dataset = utils.make_or_load_from_cache(
        "one_cue_dataset",
        dataset_config,
        make_dataset_labelled_according_to_cue,
        cache_path=CACHE_PATH
    )

    diag_and_off_diag_dataset = DatasetWithDiagAndOffDiagCells(
        easy_to_bias_cue_dataset,
        ground_truth_cue_dataset,
        off_diag_samples_proportion,
        unlabelled_off_diag=unlabelled_off_diag,
        equal_diag_off_diag_in_batch=equal_diag_off_diag_in_batch,
        shuffle=(shuffle if equal_diag_off_diag_in_batch else False),
        dataset_size=dataset_size,
        off_diag_multiplier=off_diag_multiplier
    )

    # if off_diag_multiplier is not None:
    #     assert off_diag_multiplier > 1
    #     ######### ATTENTION PLEASE
    #     pass  # please put your code instead of this line
    #     ######### THANK YOU FOR YOUR ATTENTION

    #     ######### ANSWER BEGIN

    #     for i in range(len(diag_and_off_diag_dataset.triplet_indices)):
    #         triplet_id = diag_and_off_diag_dataset.triplet_indices[i]
    #         triplet = diag_and_off_diag_dataset.triplets[triplet_id]

    #         if triplet[1] != triplet[2]:
    #             for i in range(off_diag_multiplier - 1):
    #                 diag_and_off_diag_dataset.triplet_indices.append(triplet_id)

    #     ######### ANSWER END

    # return torch.utils.data.DataLoader(
    #     diag_and_off_diag_dataset,
    #     # batch_size=BATCH_SIZE,
    #     batch_size=batch_size,
    #     # shuffle=shuffle
    #     shuffle=(False if equal_diag_off_diag_in_batch else shuffle)
    # )

    return torch.utils.data.DataLoader(
        diag_and_off_diag_dataset,
        # batch_size=BATCH_SIZE,
        batch_size=batch_size,
        # shuffle=shuffle
        shuffle=(False if equal_diag_off_diag_in_batch else shuffle)
    )

    # if split == "test":


    # return {
    #     "diag": torch.utils.data.DataLoader(
    #         diag_and_off_diag_dataset,
    #         # batch_size=BATCH_SIZE,
    #         batch_size=batch_size,
    #         # shuffle=shuffle
    #         shuffle=(False if equal_diag_off_diag_in_batch else shuffle)
    #     )
    # }

def make_diag_off_diag_dataloder(diag_off_diag_dataloader_config):
    return get_diag_and_off_diag_dataloader(
        diag_off_diag_dataloader_config["dsprites_holder"],
        diag_off_diag_dataloader_config["num_classes"],
        diag_off_diag_dataloader_config["easy_to_bias_cue"],
        diag_off_diag_dataloader_config["ground_truth_cue"],
        diag_off_diag_dataloader_config["off_diag_samples_proportion"],
        diag_off_diag_dataloader_config["split"],
        diag_off_diag_dataloader_config["dataset_size"],
        diag_off_diag_dataloader_config["shuffle"],
        diag_off_diag_dataloader_config["batch_size"],
        diag_off_diag_dataloader_config["off_diag_multiplier"],
        diag_off_diag_dataloader_config["unlabelled_off_diag"],
        diag_off_diag_dataloader_config["equal_diag_off_diag_in_batch"]
    )

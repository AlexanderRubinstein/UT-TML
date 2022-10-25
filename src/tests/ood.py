from tests.utils import test_wrapper
import numpy as np


# MAX_STD_FOR_FREQ = 8
MAX_REL_STD_FOR_FREQ = 0.002


@test_wrapper
def test_split(dsprites_holder):

    def count_frequencies(indices):
        freqs = {}
        for idx in indices:
            latents_values_combination \
                = dsprites_holder._index_to_values_combination(idx)
            for latent_id, latent_value in enumerate(
                latents_values_combination
            ):
                freqs[latent_id] = freqs.setdefault(latent_id, {})
                freqs[latent_id][latent_value] \
                    = freqs[latent_id].setdefault(latent_value, 0) + 1

        return freqs

    def assert_similar_frequencies(freqs, latents_sizes):
        for latent_id, latent_values in freqs.items():
            values_as_array = np.array(list(latent_values.values()))
            mean = np.mean(values_as_array)
            std = np.std(values_as_array)
            if float(std / mean) >= MAX_REL_STD_FOR_FREQ:
                print("HERE")  # debug
                print("std", std)
                print(float(std / mean))
            # assert std < MAX_STD_FOR_FREQ
            assert float(std / mean) < MAX_REL_STD_FOR_FREQ
            assert len(values_as_array) == latents_sizes[latent_id]

    num_train_indices = len(dsprites_holder.indices["train"])
    num_test_indices = len(dsprites_holder.indices["test"])

    assert num_train_indices + num_test_indices == dsprites_holder.total_num_images
    assert abs(
        num_train_indices / (num_train_indices + num_test_indices)
            - dsprites_holder.train_val_split
    ) < 1e-4
    frequencies_for_train = count_frequencies(dsprites_holder.indices["train"])
    frequencies_for_test = count_frequencies(dsprites_holder.indices["test"])

    assert_similar_frequencies(
        frequencies_for_train,
        dsprites_holder.latents_sizes
    )
    assert_similar_frequencies(
        frequencies_for_test,
        dsprites_holder.latents_sizes
    )

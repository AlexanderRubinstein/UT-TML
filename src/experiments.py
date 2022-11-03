from IPython.display import clear_output
from typing import (
    Callable
)
import torch


# local modules
import train
import utils


def generic_experiment(
    n_epochs: int,
    make_train_dataloader: Callable,
    make_val_dataloaders: Callable,
    make_test_dataloaders: Callable,
    make_model: Callable,
    make_metric: Callable,
    make_criterion: Callable,
    make_optimizer: Callable,
    make_scheduler: Callable,
    do_train_func: Callable,
    random_seed: int,
    stop_after_epoch: int
) -> torch.nn.Module:
    """
    Run a generic experiment that consists of the following steps:
    make a model, a train_dataloader, val_dataloaders and test_dataloaders.
    Call local function "train.train_eval_loop" for the model
    on the train_dataloader and all val_dataloaders.
    Then call local function "train.eval_model_on_test" for the model
    on all test_dataloaders.

    Args:

        n_epochs (int): an argument for local function "train.train_eval_loop".

        make_train_dataloader (Callable): a factory function
            used to make a train_dataloader.

        make_val_dataloaders (Callable): a factory function
            used to make val_dataloaders.

        make_test_dataloaders (Callable): a factory function
            used to make test_dataloaders.

        make_model (Callable): a factory function
            used to make a model.

        make_metric, make_criterion, make_optimizer,
        make_scheduler, do_train_func, random_seed,
        stop_after_epoch: arguments
            for local function "train.train_eval_loop"

    Returns:

        model (torch.nn.Module): the model created and trained by this function.
    """

    utils.apply_random_seed(random_seed)

    model = make_model()

    train_dataloader = make_train_dataloader(model)
    val_dataloaders = make_val_dataloaders(model)
    test_dataloaders = make_test_dataloaders(model)

    train.train_eval_loop(
        model,
        train_dataloader,
        val_dataloaders,
        n_epochs=n_epochs,
        make_metric=make_metric,
        make_criterion=make_criterion,
        make_optimizer=make_optimizer,
        make_scheduler=make_scheduler,
        do_train_func=do_train_func,
        random_seed=random_seed,
        stop_after_epoch=stop_after_epoch,
        clear_output_func=clear_output
    )

    train.eval_model_on_test(model, test_dataloaders, make_metric=make_metric)

    return model

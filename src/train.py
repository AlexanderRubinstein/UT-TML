import torch
import random
import numpy as np
import torch.optim as optim
from torchmetrics import Accuracy
from IPython.display import clear_output
from typing import (
    Dict,
    List,
    Callable,
    Tuple
)


# local modules
import utils
from utils import (
    show_images_batch,
    make_named_labels_batch
)


ROUND_TO = 3
LOG_SEPARATOR = "=" * 80
RANDOM_SEED = 42


# sgd optimizer params
SGD_MOMENTUM = 0.9


# adam optimizer params
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0


DECAY_PARAM = 0.9


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    is_train: bool,
    compute_metric: Callable,
    msg_prefix: str,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    optimizer: object = None,
    criterion: Callable = None,
    do_train_func: Callable = None,
    show_random_batch_with_predictions: bool = False
) -> None:
    """
    Dataloader's inputs are forwarded through the model to get predictions.
    Then dataloader's labels are used for metrics computation.
    If the mode is "train" then in addition to that by using <do_train_func>
    losses are computed and logged and model weights are updated.

    Args:

        model (torch.nn.Module): a model which predictions for the inputs
            are computed and weights are updated if <is_train> is True.

        dataloader (torch.utils.data.DataLoader): a dataloader
            which generates inputs and labels.

        is_train (bool): a flag, if it is True, then the mode is "train",
            otherwise the mode is "eval".

        compute_metric (Callable): a function to compute metrics
            using <model>'s predictions and labels.

        msg_prefix (str): a message prefix to write before logs in the stdout.

        device (torch.device): a device to forward model
            and compute metrics and losses on.
            Default: torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        optimizer, criterion: args for the <do_train_func>
            (same as for local function "do_default_train_func").
            Default: None for all

        do_train_func (Callable): a function that defines
            how to accumulate loss gradients w.r.t. model weights
            and make optimizer steps if <is_train> is True.
            Default: None

        show_random_batch_with_predictions (bool): a flag, if it is True
            random batch from the <dataloader> will be shown
            using local function "utils.show_images_batch".
            Default: False
    """

    def make_epoch_stats(epoch_histories):
        epoch_stats = {}
        for stat_name, stat_epoch_history in epoch_histories.items():

            mean_stat = np.mean(
                np.array(stat_epoch_history)
            )

            if not "loss" in stat_name:
                mean_stat = round(mean_stat, ROUND_TO)

            epoch_stats[stat_name] = mean_stat

        return epoch_stats

    def print_progress(i, total_iters):
        is_last_iter = (i + 1 == total_iters)
        last_msg_char = '\r'
        print_end = ''
        msg = f"{msg_prefix} step {i + 1}/{total_iters}{last_msg_char}"
        print(msg, end=print_end, flush=True)
        if is_last_iter:
            print(" " * len(msg), flush=True)

    assert compute_metric
    compute_metric.to(device)

    if is_train:
        assert do_train_func
        assert optimizer
        assert criterion
        model.train()
    else:
        model.eval()

    total_iters = len(dataloader)
    metric_name = str(compute_metric).split("()")[0]

    model.to(device)

    if show_random_batch_with_predictions:
        sampled_batch = int(random.random() * total_iters)

    epoch_histories = {}

    for i, dataloader_items in enumerate(dataloader):

        assert len(dataloader_items) >= 2

        images_batch = dataloader_items[0]
        labels_batch = dataloader_items[1]

        images_batch = images_batch.to(device)

        if isinstance(labels_batch, list):
            assert not is_train
            assert len(dataloader_items) == 2
            for j in range(len(labels_batch)):
                labels_batch[j] = labels_batch[j].to(device)
        else:
            labels_batch = labels_batch.to(device)

        second_labels_batch = None

        if len(dataloader_items) > 2:
            second_labels_batch = dataloader_items[2]
            assert torch.is_tensor(second_labels_batch)
            second_labels_batch = second_labels_batch.to(device)

        if is_train:

            assert do_train_func
            assert criterion
            assert optimizer

            pred_batch, epoch_histories = do_train_func(
                model,
                criterion,
                optimizer,
                images_batch,
                labels_batch,
                second_labels_batch,
                epoch_histories
            )

        else:
            with torch.no_grad():
                pred_batch = model(images_batch)

        if isinstance(labels_batch, list):
            for cue_name, single_label in zip(
                dataloader.cue_names,
                labels_batch
            ):
                metric = compute_metric(
                    pred_batch,
                    single_label
                )
                utils.append_to_list_in_dict(
                    epoch_histories,
                    f"{metric_name} for {cue_name}",
                    metric.sum().item()
                )
        else:
            metric = compute_metric(
                pred_batch,
                labels_batch
            )
            utils.append_to_list_in_dict(
                epoch_histories,
                metric_name,
                metric.sum().item()
            )

        if show_random_batch_with_predictions and i == sampled_batch:

            if isinstance(labels_batch, list):
                assert hasattr(dataloader, "cue_names")
                label_names = dataloader.cue_names
            else:
                label_names = ["label"]

            named_labels_batch = make_named_labels_batch(
                label_names,
                labels_batch
            )
            named_labels_batch["prediction"] = pred_batch.argmax(-1).cpu()
            show_images_batch(images_batch, named_labels_batch)

        print_progress(i, total_iters)

    epoch_stats = make_epoch_stats(epoch_histories)

    return epoch_stats


def train_eval_loop(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloaders: Dict[str, torch.utils.data.DataLoader],
    n_epochs: int,
    make_metric: Callable,
    make_criterion: Callable,
    make_optimizer: Callable,
    make_scheduler: Callable,
    do_train_func: Callable,
    random_seed: int = RANDOM_SEED,
    stop_after_epoch: int = None,
    clear_output_func: Callable = clear_output
) -> None:
    """
    Run a train and eval loop for a model and print values
    and plot graphs for metrics and losses.
    Within one loop iteration (aka epoch) local function "run_epoch"
    is called for the model on train_dataloader in a "train" mode
    and on each of val_dataloaders in an "eval" mode.

    Args:

        model (torch.nn.Module): a model to train on <train_dataloader>
            and eval on <val_dataloaders>.

        train_dataloader (torch.utils.data.DataLoader): a dataloader to train
            the <model> on.

        val_dataloaders (Dict[str, torch.utils.data.DataLoader]): a dictionary
            that maps validation dataloader name
            to the validation dataloader to eval the <model> on.

        n_epochs (int): number of iterations in the train and eval loop.

        make_metric (Callable): a factory function used for making metrics.

        make_criterion (Callable): a factory function
            used for making a criterion aka loss.

        make_optimizer (Callable): a factory function
            used for making an optimizer of the <model> weights.

        make_scheduler (Callable): a factory function
            used for making a scheduler for the <optimizer>'s learning rate.

        do_train_func (Callable): arg for local function "run_epoch".

        random_seed (int): a random seed
            used to init random number generator.
            Default: RANDOM_SEED

        stop_after_epoch (int): the number of the iteration
            after which the loop should be forced to stop.
            Useful when changing <n_epochs> affects some other parameters
            and should be fixed but the loop needs to be stopped
            after a certain step smaller than <n_epochs>.
            If it is None, this argument is ignored.
            Default: None

        clear_output_func (Callable): a function used for clearing output.
            Useful for IPython notebook.
            Default: IPython.display.clear_output
    """


    def log_stats(stats_history, msg_prefix, plots_name):
        print(LOG_SEPARATOR, flush=True)
        print(f"{msg_prefix} mean stats:", flush=True)
        for stat_name, stat_history in stats_history.items():
            print(f"    {stat_name}: {stat_history[-1]}", flush=True)
        utils.plot_stats(stats_history, plots_name)
        print(LOG_SEPARATOR, flush=True)

    def update_attr_if_exists(obj, attr, new_value):
        if hasattr(obj, attr):
            setattr(obj, attr, new_value)

    utils.apply_random_seed(random_seed)

    train_stats = {}
    val_stats = {}
    compute_metric = make_metric()
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    for epoch in range(n_epochs):

        msg_prefix = f"Epoch {epoch + 1}/{n_epochs}"

        train_msg_prefix = msg_prefix + " train"
        val_msg_prefix = msg_prefix + " val"

        training_progress = float(epoch / (n_epochs - 1)) if n_epochs > 1 else 0

        update_attr_if_exists(criterion, "training_progress", training_progress)

        update_attr_if_exists(scheduler, "training_progress", training_progress)

        train_epoch_stats = run_epoch(
            model,
            train_dataloader,
            is_train=True,
            compute_metric=compute_metric,
            msg_prefix=train_msg_prefix,
            optimizer=optimizer,
            criterion=criterion,
            do_train_func=do_train_func
        )

        if clear_output_func:
            clear_output_func()

        utils.append_dict(train_stats, train_epoch_stats)
        log_stats(
            train_stats,
            train_msg_prefix,
            "train_stats"
        )

        for val_dataloader_name, val_dataloader in val_dataloaders.items():

            if val_dataloader_name not in val_stats:
                val_stats[val_dataloader_name] = {}

            current_val_msg_prefix = \
                f"{val_msg_prefix} {val_dataloader_name}"

            test_epoch_stats = run_epoch(
                model,
                val_dataloader,
                is_train=False,
                compute_metric=compute_metric,
                msg_prefix=current_val_msg_prefix
            )

            utils.append_dict(
                val_stats[val_dataloader_name],
                test_epoch_stats
            )

            log_stats(
                val_stats[val_dataloader_name],
                current_val_msg_prefix,
                "val_stats"
            )

        scheduler.step()

        if stop_after_epoch is not None and stop_after_epoch == epoch:
            break


def eval_model_on_test(
    model: torch.nn.Module,
    test_dataloaders: Dict[str, torch.utils.data.DataLoader],
    make_metric: Callable,
    show_random_batch_with_predictions: bool = True
) -> None:
    """
    Evaluate a model on given dataloaders, i.e. call local "run_epoch" function
    for the model on each of test dataloaders in an "eval" mode.

    Args:

        model (torch.nn.Module): a model to eval on <test_dataloaders>.

        test_dataloaders (Dict[str, torch.utils.data.DataLoader]): a dictionary
            that maps names of dataloaders on which the <model> is evaluated
            to that very dataloaders .

        make_metric (Callable): a factory function for making metrics.

        show_random_batch_with_predictions: arg
            for the local function "run_epoch".
            Default: True
    """
    compute_metric = make_metric()
    for test_dataloader_name, test_dataloader in test_dataloaders.items():
        msg_prefix = f"Test on {test_dataloader_name}"
        epoch_stats = run_epoch(
            model,
            test_dataloader,
            compute_metric=compute_metric,
            is_train=False,
            msg_prefix=msg_prefix,
            show_random_batch_with_predictions\
                =show_random_batch_with_predictions
        )
        for stat_name, stat_value in epoch_stats.items():
            print(f"{msg_prefix} {stat_name}: {stat_value}")


def make_accuracy():
    return Accuracy()


def make_ce_criterion():
    return torch.nn.CrossEntropyLoss()


def prepare_sgd_optimizer_maker(start_lr, momentum=SGD_MOMENTUM):

    def make_sgd_optimizer(model):
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=start_lr,
            momentum=momentum
        )

    return make_sgd_optimizer


def prepare_adam_optimizer_maker(
    start_lr,
    betas=ADAM_BETAS,
    weight_decay=ADAM_WEIGHT_DECAY,
    eps=ADAM_EPS
):

    def make_adam_optimizer(model):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=start_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

    return make_adam_optimizer


def prepare_exp_scheduler_maker(gamma=DECAY_PARAM):

    def make_exp_scheduler(optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    return make_exp_scheduler


def do_default_train_func(
    model: torch.nn.Module,
    criterion: Callable,
    optimizer: object,
    images_batch: torch.tensor,
    labels_batch: torch.tensor,
    second_labels_batch: torch.tensor,
    epoch_histories: Dict[str, List[float]]
) -> Tuple[torch.tensor, Dict[str, List[float]]]:

    """
    <do_train_func> for local function "run_epoch" in a standard scenario.

    Args:

        model (torch.nn.Module): a model which weights should be updated.

        criterion (Callable): a function that computes loss.

        optimizer (optimizer class): an optimizer that will update
            the <model>'s weights using gradients of the <criterion>.

        images_batch (torch.tensor): a batch of image inputs to the model.

        labels_batch (torch.tensor): a batch of labels for the <images_batch>.

        second_labels_batch (torch.tensor): a batch of additional labels
            for the <images_batch>, makes sense only for special types
            of losses.

        epoch_histories (Dict[str, List[float]]): a dictionary
            that maps metrics and losses names
            to their values throughout different epochs.

    Returns:

        a tuple of

            pred_batch (torch.tensor): predictions made by the model
                for the <images_batch>.

            epoch_histories (Dict[str, List[float]]): same as in the Args,
                but with new metrics and losses values
                appened to the corresponding lists.
    """

    pred_batch = model(images_batch)

    if (
        hasattr(criterion, "use_second_labels")
            and criterion.use_second_labels
    ):

        loss = criterion(pred_batch, labels_batch, second_labels_batch)

    else:

        loss = criterion(pred_batch, labels_batch)

    utils.append_to_list_in_dict(epoch_histories, "loss", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return pred_batch, epoch_histories

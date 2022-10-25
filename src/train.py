import torch
import random
import numpy as np

# local modules
import utils
from datasets import EMPTY_LABEL
from utils import (
    show_images_batch,
    make_named_labels_batch
)


ROUND_TO = 3
# EMPTY_LABEL = -1
LOG_SEPARATOR = "=" * 80
RANDOM_SEED = 42


def run_epoch(
    model,
    dataloader,
    is_train,
    compute_metric,
    msg_prefix,
    # do_train_func,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    optimizer=None,
    criterion=None,
    # do_train_func=do_default_train_func
    do_train_func=None,
    show_random_batch_with_predictions=False
):

    # def prepare_pred_label(pred, label, diag_mask):
    def prepare_pred_label(pred, label):
        pred = pred[0] if isinstance(pred, tuple) else pred
        # if diag_mask is not None:
        #     assert not isinstance(label, list)
        #     pred = pred[diag_mask]
        #     label = label[diag_mask]
        return pred, label

    def make_epoch_stats(epoch_histories):
        epoch_stats = {}
        for stat_name, stat_epoch_history in epoch_histories.items():
            mean_stat = np.mean(
                np.array(stat_epoch_history)
            )
            if not "loss" in stat_name:
                mean_stat = round(mean_stat, ROUND_TO)
            epoch_stats[stat_name] = np.mean(
                mean_stat
            )
        return epoch_stats

    def print_progress(i, total_iters):
        is_last_iter = (i + 1 == total_iters)
        last_msg_char = '\r'
        print_end = ''
        msg = f"{msg_prefix} step {i + 1}/{total_iters}{last_msg_char}"
        print(
            msg,
            end=print_end
        )
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
    # metric_sum = 0
    batches_with_skipped_metric = 0
    metric_name = str(compute_metric).split("()")[0]

    model.to(device)

    if show_random_batch_with_predictions:
        sampled_batch = int(random.random() * total_iters)

    epoch_histories = {}

    for i, dataloader_items in enumerate(dataloader):
        assert len(dataloader_items) >= 2
        image = dataloader_items[0]
        label = dataloader_items[1]

        image = image.to(device)

        if isinstance(label, list):
            assert not is_train
            assert len(dataloader_items) == 2
            for j in range(len(label)):
                label[j] = label[j].to(device)
        else:
            label = label.to(device)

        second_label = None
        # diag_mask = None

        if len(dataloader_items) > 2:
            second_label = dataloader_items[2]
            second_label = second_label.to(device)

        if is_train:
            # pred, epoch_histories, diag_mask = do_train_func(
            pred, epoch_histories = do_train_func(
                model,
                criterion,
                optimizer,
                image,
                label,
                second_label,
                epoch_histories
            )
        else:
            with torch.no_grad():
                pred = model(image)
            # if not isinstance(label, list) and label.min() == EMPTY_LABEL:
            #     diag_mask = torch.isclose(label, second_label)

        # original_pred = pred  # oldebug

        # pred, label = prepare_pred_label(pred, label, diag_mask)
        pred, label = prepare_pred_label(pred, label)

        # if pred.shape[0] == 0:
        #     # if all images are off-diag, do not compute metric for them
        #     assert (~diag_mask).all()
        #     batches_with_skipped_metric += 1

        # else:

        if isinstance(label, list):
            for cue_name, single_label in zip(dataloader.cue_names, label):
                metric = compute_metric(
                    pred,
                    single_label
                )
                utils.append_to_list_in_dict(
                    epoch_histories,
                    f"{metric_name} for {cue_name}",
                    metric.sum().item()
                )
        else:
            metric = compute_metric(
                    pred,
                    label
                )
            utils.append_to_list_in_dict(
                epoch_histories,
                metric_name,
                metric.sum().item()
            )

        if show_random_batch_with_predictions and i == sampled_batch:
            if isinstance(label, list):
                assert hasattr(dataloader, "cue_names")
                label_names = dataloader.cue_names
            else:
                label_names = ["label"]
            labels_batch = make_named_labels_batch(label_names, label)
            labels_batch["prediction"] = pred.argmax(-1).cpu()
            show_images_batch(image, labels_batch)

        # if not is_train and second_label is not None and i == sampled_batch:

        #     metric_y = compute_metric(
        #         # extract_pred(pred, diag_mask),
        #         original_pred[0],
        #         second_label
        #     )
        #     metric_d = compute_metric(
        #         # extract_pred(pred, diag_mask),
        #         original_pred[1],
        #         second_label
        #     )
        #     print("metric_y:", metric_y)
        #     print("metric_d:", metric_d)

        #     image_batch = image.cpu()
        #     if diag_mask is not None:
        #         # label = label[diag_mask]
        #         diag_image_batch = image_batch[diag_mask]
        #         off_diag_image_batch = image_batch[~diag_mask]
        #         if diag_image_batch.shape[0] > 0:
        #             print("diag image, label:")
        #             utils.show_images_batch(diag_image_batch, label.cpu())
        #         else:
        #             assert off_diag_image_batch.shape[0] == diag_mask.shape[0]
        #             print("diag is empty for this batch")
        #         if off_diag_image_batch.shape[0] > 0:
        #             print("off_diag image, second label:")
        #             utils.show_images_batch(off_diag_image_batch, second_label[~diag_mask].cpu())
        #         else:
        #             assert diag_image_batch.shape[0] == diag_mask.shape[0]
        #             print("off_diag is empty for this batch")

        #     else:

        #         print("image, label:")
        #         utils.show_images_batch(image_batch, label.cpu())

        #     print("image, second label:")
        #     utils.show_images_batch(image_batch, second_label.cpu())
        #     print("image, Gy pred:")
        #     utils.show_images_batch(image_batch, original_pred[0].argmax(-1).cpu())
        #     print("image, Gd pred:")
        #     utils.show_images_batch(image_batch, original_pred[1].argmax(-1).cpu())  # till clear_output - olddebug

        print_progress(i, total_iters)

    epoch_stats = make_epoch_stats(epoch_histories)

    return epoch_stats


# def train_eval_loop(
#     model,
#     train_dataloader,
#     test_dataloaders,
#     n_epochs,
#     make_metric=make_accuracy,
#     make_criterion=make_ce_criterion,
#     make_optimizer=make_sgd_optimizer(),
#     make_scheduler=make_exp_scheduler(DECAY_PARAM),
#     do_train_func=do_default_train_func,
#     random_seed=RANDOM_SEED,
#     stop_after_epoch=None
# ):
def train_eval_loop(
    model,
    train_dataloader,
    test_dataloaders,
    n_epochs,
    make_metric,
    make_criterion,
    make_optimizer,
    make_scheduler,
    do_train_func,
    random_seed=RANDOM_SEED,
    stop_after_epoch=None,
    clear_output_func=None
):
    def log_stats(stats_history, msg_prefix, plots_name):
        print(LOG_SEPARATOR, flush=True)
        print(f"{msg_prefix} mean stats:")
        for stat_name, stat_history in stats_history.items():
            print(f"    {stat_name}: {stat_history[-1]}")
        utils.plot_stats(stats_history, plots_name)
        print(LOG_SEPARATOR, flush=True)

    def update_attr(obj, attr, new_value):
        if hasattr(obj, attr):
            setattr(obj, attr, new_value)

    utils.apply_random_seed(random_seed)

    train_stats = {}
    test_stats = {}
    compute_metric = make_metric()
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    for epoch in range(n_epochs):

        msg_prefix = f"Epoch {epoch + 1}/{n_epochs}"

        train_msg_prefix = msg_prefix + " train"
        test_msg_prefix = msg_prefix + " test"

        training_progress = float(epoch / (n_epochs - 1)) if n_epochs > 1 else 0
        update_attr(criterion, "training_progress", training_progress)
        update_attr(scheduler, "training_progress", training_progress)
        if hasattr(criterion, "training_progress"):
            criterion.update_f_lambda()

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

        for test_dataloader_name, test_dataloader in test_dataloaders.items():
            if test_dataloader_name not in test_stats:
                test_stats[test_dataloader_name] = {}
            current_test_msg_prefix = \
                f"{test_msg_prefix} {test_dataloader_name}"
            test_epoch_stats = run_epoch(
                model,
                test_dataloader,
                is_train=False,
                compute_metric=compute_metric,
                msg_prefix=current_test_msg_prefix
            )
            utils.append_dict(
                test_stats[test_dataloader_name],
                test_epoch_stats
            )
            log_stats(
                test_stats[test_dataloader_name],
                current_test_msg_prefix,
                "test_stats"
            )

        scheduler.step()

        if stop_after_epoch is not None and stop_after_epoch == epoch:
            break


def eval_model_on_test(
    model,
    test_dataloaders,
    make_metric,
    show_random_batch_with_predictions=True
):
    for test_name in test_dataloaders:
        msg_prefix = f"Test on {test_name}"
        epoch_stats = run_epoch(
            model,
            test_dataloaders[test_name],
            compute_metric=make_metric(),
            is_train=False,
            msg_prefix=msg_prefix,
            show_random_batch_with_predictions\
                =show_random_batch_with_predictions
        )
        for stat_name, stat_value in epoch_stats.items():
            print(f"{msg_prefix} {stat_name}: {stat_value}")

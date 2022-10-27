from IPython.display import clear_output


# local modules
import train
import utils


def generic_experiment(
    n_epochs,
    make_train_dataloader,
    make_val_dataloaders,
    make_test_dataloaders,
    make_model,
    make_metric,
    make_criterion,
    make_optimizer,
    make_scheduler,
    do_train_func,
    random_seed,
    stop_after_epoch
):

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

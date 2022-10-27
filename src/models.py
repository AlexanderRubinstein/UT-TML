from torchvision import models
from torchvision.models.resnet import (
    ResNet50_Weights,
    ResNet18_Weights
)
import torch.nn as nn


# local modules
from datasets import (
    NUM_CLASSES,
    N_COLORS
)


RESNET_DEFAULT_N_CHANNELS = 3
RESNET_DEFAULT_N_CLASSES = 1000


def build_resnet18(pretrained, n_classes, n_channels):
    return build_resnet_n(
        constructor=models.resnet18,
        default_weights=ResNet18_Weights.DEFAULT,
        pretrained=pretrained,
        n_classes=n_classes,
        n_channels=n_channels
    )


def build_resnet50(pretrained, n_classes, n_channels):
    return build_resnet_n(
        constructor=models.resnet50,
        default_weights=ResNet50_Weights.DEFAULT,
        pretrained=pretrained,
        n_classes=n_classes,
        n_channels=n_channels
    )


def build_resnet_n(
    constructor,
    default_weights,
    pretrained,
    n_classes,
    n_channels
):

    def update_resnet_top_layer_classes(model, n_classes):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model

    def update_resnet_first_layer_channels(model, n_channels):

        model.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias
        )

        return model

    def warning_for_random_weights(layer_num):
        return (
            "Although resnet weights are pretrained, "
            f"{layer_num} layer will be random "
            "due to non-default number of classes."
        )

    model = constructor(
        weights=(default_weights if pretrained else None)
    )

    if n_channels != RESNET_DEFAULT_N_CHANNELS:

        model = update_resnet_first_layer_channels(
            model=model,
            n_channels=n_channels
        )
        if pretrained:
            print(warning_for_random_weights("first"))

    if n_classes != RESNET_DEFAULT_N_CLASSES:

        model = update_resnet_top_layer_classes(
            model=model,
            n_classes=n_classes
        )
        if pretrained:
            print(warning_for_random_weights("last"))

    return model


def prepare_resnet18_maker(
    pretrained=False,
    n_channels=N_COLORS,
    n_classes=NUM_CLASSES
):

    def make_resnet18():
        return build_resnet18(
            pretrained=pretrained,
            n_channels=n_channels,
            n_classes=n_classes
        )

    return make_resnet18

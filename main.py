from math import ceil
import numpy as np
import torch
from torch import nn, optim, Tensor
import tensorflow as tf
from torchvision import models

from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModel

np.random.seed(1234)

num_points_per_train_split = 5000
num_points_per_test_split = 1000
loss_fn = nn.CrossEntropyLoss()
epochs = 20
batch_size = 64

num_reference_models = 32
fpr_tolerance_list = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

def preprocess_cifar100_dataset():
    input_shape, num_classes = (32, 32, 3), 100

    # split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # switch channels to meet pytorch expectations
    x_train = np.moveaxis(x_train, -1, 1)
    x_test = np.moveaxis(x_test, -1, 1)

    # convert labels into one hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes

x_train_all, y_train_all, x_test_all, y_test_all, input_shape, num_classes = preprocess_cifar100_dataset()

# create the target model's dataset
dataset = Dataset(
    data_dict={
        'train': {'x': x_train_all, 'y': y_train_all},
        'test': {'x': x_test_all, 'y': y_test_all}
    },
    default_input='x',
    default_output='y'
)

datasets_list = dataset.subdivide(
    num_splits=num_reference_models + 1,
    delete_original=True,
    in_place=True,
    return_results=True,
    method='hybrid',
    split_size={'train': num_points_per_train_split, 'test': num_points_per_test_split}
)


def get_model():
  model = models.squeezenet1_0(pretrained=False)
  model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
  model.num_classes = num_classes
  return model


def train_model(model, k, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    x = dataset.get_feature(split_name=f'train{k:03d}', feature_name='<default_input>')
    y = dataset.get_feature(split_name=f'train{k:03d}', feature_name='<default_output>')
    n_samples = x.shape[0]
    n_batches = ceil(n_samples / batch_size)
    x = Tensor(np.array_split(x, n_batches)).to(device)
    y = Tensor(np.array_split(y, n_batches)).to(device)
    for epoch in range(epochs):
        epoch_loss, acc = 0.0, 0.0
        for b in range(n_batches):
            optimizer.zero_grad()
            y_pred = model(x[b])
            loss = loss_fn(y[b], y_pred)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            acc += torch.sum(y_pred.argmax(axis=1) == y[b].argmax(axis=1))
        acc /= n_samples
        epoch_loss /= n_samples
        print(f'model #{k:02d}, epoch #{epoch:03d}:\ttrain_acc = {acc:.3f}\ttrain_loss = {epoch_loss:.3e}')
    return model


model_wrappers = [
    PytorchModel(
        model_obj=train_model(get_model(), k, 'cuda:0'),
        loss_fn=loss_fn
    )
    for k in range(num_reference_models + 1)
]

target_info_source = InformationSource(
    models=[model_wrappers[0]],
    datasets=[datasets_list[0]]
)

reference_info_source = InformationSource(
    models=model_wrappers[1:],
    datasets=datasets_list[1:]
)

audit_obj = Audit(
    metrics=MetricEnum.REFERENCE,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    target_info_sources=target_info_source,
    reference_info_sources=reference_info_source,
    fpr_tolerances=fpr_tolerance_list
)
audit_obj.prepare()
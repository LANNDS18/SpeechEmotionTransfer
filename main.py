import os

import torch

from analyzer import divide_into_source_target
from util import validate_log_dirs
from trainer import Trainer


def main():
    """ NOTE: The input is rescaled to [-1, 1] """

    dirs = validate_log_dirs()

    os.makedirs(dirs['logdir'])

    # {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
    source = [1, 2, 4, 5]
    target = 3
    source_data, target_data = divide_into_source_target(source, target)

    print("source data: ", source_data.shape)
    print("target data: ", target_data.shape)

    machine = Trainer(name='VAW1')

    machine.load_data(source_data, target_data)
    machine.train()

    source = [1, 2, 3, 5]
    target = 4
    source_data, target_data = divide_into_source_target(source, target)

    print("source data: ", source_data.shape)
    print("target data: ", target_data.shape)

    f1 = './model/model_VAW1.pt'
    model1 = torch.load(f1)

    machine = Trainer(name='VAW_2')
    machine.load_data(source_data, target_data)
    Trainer.G = model1.G
    Trainer.Encoder = model1.Encoder
    Trainer.D = model1.D
    machine.train()

    source = [1, 2, 3, 4]
    target = 5
    source_data, target_data = divide_into_source_target(source, target)

    f2 = './model/model_VAW2.pt'
    model2 = torch.load(f2)

    machine = Trainer(name='VAW_3')
    machine.load_data(source_data, target_data)
    Trainer.G = model2.G
    Trainer.Encoder = model2.Encoder
    Trainer.D = model2.D
    machine.train()


def First_Round():
    f1 = './model/model_16.pt'
    model1 = torch.load(f1)

    source = [1, 2, 3, 5]
    target = 4
    source_data, target_data = divide_into_source_target(source, target)

    machine = Trainer(name='VAW_2')
    machine.load_data(source_data, target_data)
    Trainer.G = model1.G
    Trainer.Encoder = model1.Encoder
    Trainer.D = model1.D
    machine.train()


def Second_Round():
    f1 = './model/model_16.pt'
    model1 = torch.load(f1)

    source = [1, 2, 3, 5]
    target = 4
    source_data, target_data = divide_into_source_target(source, target)

    machine = Trainer(name='VAW_2')
    machine.load_data(source_data, target_data)
    Trainer.G = model1.G
    Trainer.Encoder = model1.Encoder
    Trainer.D = model1.D
    machine.train()


def Third_Round():
    source = [1, 2, 3, 4, 6]
    target = 5
    source_data, target_data = divide_into_source_target(source, target)
    machine = Trainer(name='VAW_00')
    machine.load_data(source_data, target_data)
    machine.train()

    f1 = './model/model_VAW_00.pt'
    model1 = torch.load(f1)
    source = [1, 2, 4, 5, 6]
    target = 3
    source_data, target_data = divide_into_source_target(source, target)
    machine = Trainer(name='VAW_01')
    machine.load_data(source_data, target_data)
    Trainer.G = model1.G
    Trainer.Encoder = model1.Encoder
    Trainer.D = model1.D
    machine.train()


def all_t_4():
    f2 = './model/model_VAW_01.pt'
    model2 = torch.load(f2)
    source = [1, 2, 3, 5, 6]
    target = 4
    source_data, target_data = divide_into_source_target(source, target)
    machine = Trainer(name='VAW_All_to_4')
    machine.load_data(source_data, target_data)
    Trainer.G = model2.G
    Trainer.Encoder = model2.Encoder
    Trainer.D = model2.D
    machine.train()


if __name__ == '__main__':
    all_t_4()

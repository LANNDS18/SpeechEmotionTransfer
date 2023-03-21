import os

from analyzer import divide_into_source_target
from util import validate_log_dirs
from trainer import Trainer


# from trainer import *


def main():
    """ NOTE: The input is rescaled to [-1, 1] """

    dirs = validate_log_dirs()

    os.makedirs(dirs['logdir'])

    source = [1, 2, 3, 4]
    target = 5
    source_data, target_data = divide_into_source_target(source, target)

    print(source_data.shape)
    print(target_data.shape)

    machine = Trainer()

    machine.load_data(source_data, target_data)
    # machine.train()


if __name__ == '__main__':
    main()

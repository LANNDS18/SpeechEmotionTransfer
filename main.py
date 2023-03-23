import os

from analyzer import divide_into_source_target
from util import validate_log_dirs
from trainer import Trainer


def main():
    """ NOTE: The input is rescaled to [-1, 1] """

    dirs = validate_log_dirs()

    os.makedirs(dirs['logdir'])

    # {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
    source = [1]
    target = 5

    source_data, target_data = divide_into_source_target(source, target)

    print("source data: ", source_data.shape)
    print("target data: ", target_data.shape)

    machine = Trainer()

    machine.load_data(source_data, target_data)
    machine.train()


if __name__ == '__main__':
    main()

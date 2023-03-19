import argparse
import errno
import os

import numpy as np

from analyzer import read_whole_features


def main():
    try:
        os.makedirs('./etc')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # ==== Save max and min value ====
    x = read_whole_features(args.train_file_pattern)  # TODO: use it as a obj and keep `n_files`
    x_all = list()
    y_all = list()
    f0_all = list()

    counter = 1
    for features in x:  # TODO: read according to speaker instead of all speakers
        # print(features)

        print('\rProcessing {}: {}'.format(counter, features['filename']), end='')
        x_all.append(features['sp'])
        y_all.append(features['speaker'])
        f0_all.append(features['f0'])
        counter += 1

        print()

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)

    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]

    # ==== F0 stats ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s), flush=True)
        f0 = f0_all[SPEAKERS.index(s) == y_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()

        # Save as `float32`
        with open('./etc/{}.npf'.format(s), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())

    # ==== Min/Max value ====
    # mu = x_all.mean(0)
    # std = x_all.std(0)
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open('./etc/{}_xmin.npf'.format('vcc2016'), 'wb') as fp:
        fp.write(q005.tostring())

    with open('./etc/{}_xmax.npf'.format('vcc2016'), 'wb') as fp:
        fp.write(q995.tostring())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAW-GAN build.py')
    parser.add_argument('--corpus_name', default='vcc2016', help='Corpus Name')
    parser.add_argument('--speaker_list', default='./etc/speakers.tsv', help='Speaker list (one speaker per line)')
    parser.add_argument('--train_file_pattern', default='./dataset/vcc2016/bin/Training Set/*/*.bin',
                        help='training dir (to *.bin)')
    main()

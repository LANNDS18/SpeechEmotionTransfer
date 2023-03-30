import errno
import os

import numpy as np

from analyzer import read_whole_features, extract_and_save


def main():
    try:
        os.makedirs('./bin/etc')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # ==== Save max and min value ====
    x = read_whole_features()
    x_all = list()
    f0_all = list()
    embedding_all = list()
    emotion_all = list()

    counter = 1
    for features in x:

        print('\rProcessing {}'.format(counter), end='')
        x_all.append(features['sp'])
        f0_all.append(features['f0'])
        embedding_all.append(features['emb'])
        emotion_all.append(features['emotion'])
        counter += 1

        print()

    x_all = np.concatenate(x_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)
    emotion_all = np.concatenate(emotion_all, axis=0)
    embedding_all = np.concatenate(embedding_all, axis=0)

    EMOTION = [0, 1, 2, 3, 4, 5, 6, 7]

    # ==== F0 stats ====
    for s in EMOTION:
        print('Emotion {}'.format(s), flush=True)
        f0 = f0_all[EMOTION.index(s) == emotion_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()

        # Save as `float32`
        print('  mu: {}'.format(mu))
        print('  std: {}'.format(std))
        with open('./bin/etc/{}.npf'.format(s), 'wb') as fp:
            # pass
            fp.write(np.asarray([mu, std]).tostring())

    # ==== Min/Max value ====
    # mu = x_all.mean(0)
    # std = x_all.std(0)
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    print('q005: {}'.format(q005))
    print('q995: {}'.format(q995))

    # Save as `float32`
    with open('./bin/etc/{}_xmin.npf'.format('vcc2016'), 'wb') as fp:
        # pass
        fp.write(q005.tostring())

    with open('./bin/etc/{}_xmax.npf'.format('vcc2016'), 'wb') as fp:
        fp.write(q995.tostring())


if __name__ == '__main__':
    extract_and_save()
    main()

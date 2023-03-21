import os
from os.path import join

import librosa
import numpy as np
import pyworld as pw

from emotion_representation import load_embedding
from preprocess import read_RAVDESS_from_dir

FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + 1 + 320 + 1
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`x

EPSILON = 1e-10

EMOTION_EMBEDDING_DIM = 320

# original in data arg
f0_ceil = 500
fs = 16000


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    """ Extract WORLD feature from waveform """
    _f0, t = pw.dio(x, fs, f0_ceil=f0_ceil)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)  # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def list_full_filenames(path):
    ''' return a generator of full filenames '''
    return (
        join(path, f)
        for f in os.listdir(path)
        if not os.path.isdir(join(path, f)))


def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction '''
    sample_wave, _ = librosa.load(filename, sr=fs, mono=True, dtype=np.float64, duration=3, offset=0)
    sample_wave = sample_wave.astype(np.double)
    x = sample_wave
    features = wav2pw(x, fs, fft_size=fft_size)
    _ = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    array = np.concatenate([sp, f0], axis=1).astype(dtype)
    return array


def extract_and_save():
    RAVDESS_data, complete_embedding = load_embedding()

    res = []

    for i, file in enumerate(RAVDESS_data['Path']):
        features = extract(file)

        labels = RAVDESS_data.iloc[i]['Emotion'] * np.ones(
            [features.shape[0], 1],
            np.float32,
        )

        emb = complete_embedding[i] * np.ones(
            [features.shape[0], len(complete_embedding[i])],
            np.float32,
        )

        features = np.concatenate([features, emb, labels], 1)  # from 601, 513 to 601, 513 + 1 + 320 + 1
        res.append(features)

    # print(res[:, 513])
    np.save('ravdess_complete_feature_embedding.npy', res)

    # print(res[0].shape) == (601, 835)


def load_complete_feature_embeding():
    train = np.load('ravdess_complete_feature_embedding.npy', allow_pickle=True)
    features = []
    labels = []
    for i in range(len(train)):
        features.append(train[i][:, :-1])
        labels.append(train[i][:, -1])

    return features, labels, train


'''
    print(train[0].shape)
    print(len(labels))
    print(len(features))
    print(features[0].shape)
'''


class Tanhize(object):
    ''' Normalizing `x` to [-1, 1] '''

    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin

    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        return np.clip(x, 0., 1.) * 2. - 1.

    def backward_process(self, x):
        return (x * .5 + .5) * self.xscale + self.xmin


def read_whole_features():
    """
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    """
    total = []
    rav = read_RAVDESS_from_dir(data_path='./audio_speech_actors_01-24/')

    print('{} files found'.format(len(rav['Path'])))

    features, labels, train = load_complete_feature_embeding()

    # sp should be 1440 * 601 = 864600

    for i, file in enumerate(rav["Path"]):
        value = {}
        data = train[i].astype(np.float32)
        value['sp'] = data[:, :SP_DIM]
        value['f0'] = data[:, SP_DIM]
        value['emb'] = data[:, SP_DIM + 1: SP_DIM + 1 + EMOTION_EMBEDDING_DIM]
        value['emotion'] = labels[i]
        total.append(value)

    return total


def divide_into_source_target(source: list, target: int):
    """
    Divide the data into source and target
    :param source: list of source speaker id
    :param target: target speaker id
    :param data: list of data
    :return: source data, target data
    """
    source_data = []
    target_data = []

    corpus_name = 'vcc2016'

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/{}_xmax.npf'.format(corpus_name)),
        xmin=np.fromfile('./etc/{}_xmin.npf'.format(corpus_name)),
    )

    feature, label, train = load_complete_feature_embeding()
    for i in range(len(train)):
        if label[i][0] == target or label[i][0] in source:
            for j in range(len(train[i])):
                reshaped = train[i][j].reshape(-1, FEAT_DIM)
                feature = reshaped[:, :SP_DIM]

                if normalizer is not None:
                    feature = normalizer.forward_process(feature)

                emotion_id = reshaped[:, -1].reshape(-1, 1)
                embedding = reshaped[:, SP_DIM + 1: SP_DIM + 1 + EMOTION_EMBEDDING_DIM].reshape(-1, EMOTION_EMBEDDING_DIM)
                f0 = reshaped[:, SP_DIM].reshape(-1, 1)
                test = np.concatenate((feature, f0, embedding, emotion_id), axis=1)

                if label[i][0] == target:
                    target_data.append(test)
                else:
                    source_data.append(test)

    target_data = np.concatenate(target_data, axis=0)
    source_data = np.concatenate(source_data, axis=0)

    target_data = target_data.reshape([-1, FEAT_DIM, 1, 1])
    source_data = source_data.reshape([-1, FEAT_DIM, 1, 1])

    return source_data, target_data


if __name__ == '__main__':
    # extract_and_save()
    unseen = [5, 6]
    seen = [1, 2, 3, 4, 7, 8]
    # source, target = divide_into_source_target(seen, 6)

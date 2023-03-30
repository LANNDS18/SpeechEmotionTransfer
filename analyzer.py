import os
from os.path import join

import librosa
import numpy as np
import pyworld as pw

from emotion_representation import load_embedding
from preprocess_utils import read_RAVDESS_from_dir

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


def extract_for_test(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction '''
    sample_wave, _ = librosa.load(filename, sr=fs, mono=True, dtype=np.float64, duration=3, offset=0)
    sample_wave = sample_wave.astype(np.double)
    x = sample_wave
    features = wav2pw(x, fs, fft_size=fft_size)
    ap = features['ap']
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    array = np.concatenate([ap, en], axis=1).astype(dtype)
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
        value['path'] = file

        total.append(value)

    return total


def get_data_from_(emotion_id, des='train', test_ratio=0.2):
    train_ratio = 1.0 - test_ratio
    rav = read_RAVDESS_from_dir(data_path='./audio_speech_actors_01-24/')
    rav_emo = rav[rav['Emotion'] == emotion_id]
    index = np.round(rav_emo.shape[0] * train_ratio).astype(int)
    if des == 'train':
        res = rav_emo[0:index]
    else:
        res = rav_emo[index:]
    return res


def map_to_target_path(path, target_id):
    # Split the file path into components
    path_parts = path.split('/')
    # Get the file name (the last part of the path)
    file_name = path_parts[-1]
    # Split the file name into sections based on the '-' delimiter
    name_parts = file_name.split('-')
    # Replace the third section with the new value, formatted as '0x'
    name_parts[2] = f'0{target_id}'
    # Reassemble the file name with the updated section
    new_file_name = '-'.join(name_parts)
    # Reassemble the file path with the updated file name
    new_path = '/'.join(path_parts[:-1] + [new_file_name])
    return new_path


def load_test_data(source: int, target: int, test_train_ratio=0.2):
    feature, label, data = load_complete_feature_embeding()
    test_data = get_data_from_(source, 'test', test_train_ratio)
    test_ids = test_data.index

    data = data[test_ids]

    test_res = []

    for i, d in enumerate(test_data.iloc):
        path = test_data.iloc[i]['Path']
        target_path = map_to_target_path(path, target)
        ap_en = extract_for_test(path)
        res = {'Path': test_data.iloc[i]['Path'], 'Emotion': source, 'Feature': data[i], 'Target_Path': target_path, 'ap_en': ap_en}
        test_res.append(res)

    return test_res


def divide_into_source_target(source: list, target: int, train_test_ratio=0.2, statement='01'):
    """
    Divide the data into source and target
    :param source: list of source speaker id
    :param target: target speaker id
    :param train_test_ratio:
    :param statement: '01' or '02' for RAVDESS
    :return: source train data, target data
    """
    target_data = []
    source_train_data = []

    corpus_name = 'vcc2016'

    normalizer = Tanhize(
        xmax=np.fromfile('./bin/etc/{}_xmax.npf'.format(corpus_name)),
        xmin=np.fromfile('./bin/etc/{}_xmin.npf'.format(corpus_name)),
    )

    feature, label, data = load_complete_feature_embeding()
    RAVDESS_data = read_RAVDESS_from_dir(data_path='./audio_speech_actors_01-24/')

    train_ids = []
    for i in range(len(source)):
        train_ids.append(get_data_from_(source[i], 'train', train_test_ratio).index)

    target_ids = get_data_from_(target, 'train', train_test_ratio).index

    count = 0

    for i in range(len(data)):
        path = RAVDESS_data.iloc[i]['Path']
        # Split the file path into components
        path_parts = path.split('/')
        # Get the file name (the last part of the path)
        file_name = path_parts[-1]
        # Split the file name into sections based on the '-' delimiter
        name_parts = file_name.split('-')

        if statement:
            if name_parts[4] != statement:
                continue

        if label[i][0] == target or label[i][0] in source:
            for j in range(len(data[i])):
                reshaped = data[i][j].reshape(-1, FEAT_DIM)
                feature = reshaped[:, :SP_DIM]

                if normalizer is not None:
                    feature = normalizer.forward_process(feature)

                emotion_id = reshaped[:, -1].reshape(-1, 1)
                embedding = reshaped[:, SP_DIM + 1: SP_DIM + 1 + EMOTION_EMBEDDING_DIM].reshape(-1,
                                                                                                EMOTION_EMBEDDING_DIM) * 10
                f0 = reshaped[:, SP_DIM].reshape(-1, 1)
                test = np.concatenate((feature, f0, embedding, emotion_id), axis=1)

                if label[i][0] == target and i in target_ids:
                    target_data.append(test)
                else:
                    for k in range(len(source)):
                        if i in train_ids[k]:
                            source_train_data.append(test)

        if label[i][0] in source:
            count += 1

    print(f"Number Train Source Files {len(source_train_data) / 601}, out of {count}")
    target_data = np.concatenate(target_data, axis=0)
    source_train_data = np.concatenate(source_train_data, axis=0)

    target_data = target_data.reshape([-1, FEAT_DIM, 1, 1])
    source_train_data = source_train_data.reshape([-1, FEAT_DIM, 1, 1])

    print(source_train_data.shape)

    return source_train_data, target_data


def pw2wav(features, feat_dim=513, fs=16000):
    """ NOTE: Use `order='C'` to ensure Cython compatibility """
    # print(type(features['sp']))
    # print(type(features['en']))
    en = np.reshape(features['en'], [-1, 1])
    sp = np.power(10., features['sp'])
    sp = en * sp
    if isinstance(features, dict):
        return pw.synthesize(
            features['f0'].squeeze().astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim * 2]
    f0 = features[:, feat_dim * 2]
    en = features[:, feat_dim * 2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )


if __name__ == '__main__':
    # extract_and_save()
    unseen = [5, 6]
    seen = [1]
    train_source, train_target = divide_into_source_target(seen, 6)
    print(train_source.shape)
    print(train_target.shape)
    # load_test_data(6, 2)

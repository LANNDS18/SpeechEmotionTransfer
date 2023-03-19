import glob
import os
from os.path import join

import librosa
import numpy as np
import pyworld as pw

from preprocess import read_RAVDESS_from_dir
from emotion_representation import load_embedding

FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 256 + 1
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`x

EPSILON = 1e-10

# original in data arg
f0_ceil = 500
fs = 16000


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    """ Extract WORLD feature from waveform """
    _f0, t = pw.dio(x, fs, f0_ceil == f0_ceil)  # raw pitch extractor
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
    x, _ = librosa.load(filename, sr=fs, mono=True, dtype=np.float64, duration=3, offset=0)
    features = wav2pw(x, fs, fft_size=fft_size)
    _ = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    array = np.concatenate([sp, f0], axis=1).astype(dtype)
    return array


def sample_and_extract():
    data = read_RAVDESS_from_dir(data_path='./audio_speech_actors_01-24/')
    sample = data.sample(n=1)['Path'].values[0]
    sample_wave = extract(sample, fft_size=FFT_SIZE, dtype=np.float32)
    print(sample_wave.shape)


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

        features = np.concatenate([features, emb, labels], 1)  # from 601, 513 to 601, 513 + 1 + 1 + 320
        res.append(features)

    np.save('ravdess_complete_feature_embedding.npy', res)

    print(res[0].shape)


def load_complete_feature_embeding():
    train = np.load('ravdess_complete_feature_embedding.npy', allow_pickle=True)
    features = []
    labels = []
    for i in range(len(train)):
        features.append(train[i][:, :-1])
        labels.append(train[i][:, -1])

    print(train[0].shape)
    print(len(labels))
    print(len(features))
    print(features[0].shape)

    return features, labels, train


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


#TODO: apply it to the loaded data
def read(
        file_pattern,
        data_format='NCHW',
        normalizer=None,
):
    '''
    Read only `sp` and `speaker`
    Return:
        `feature`: [b, c]
        `speaker`: [b,]
    '''

    files = [file_name for file_name in glob.glob(file_pattern)]  # 1620(only including Testing Set)

    # filename_queue = tf.train.string_input_producer(files)
    total_sp_speaker = []
    total_speaker = []
    for file_name in files:
        with open(file_name, "rb") as reader:
            bytes_data = reader.read()
            value = np.fromstring(bytes_data, dtype=np.float32).reshape([-1, FEAT_DIM])
            # print('1: ',value.shape)

            feature = value[:, :SP_DIM]  # NCHW format
            # print(feature.shape)
            if normalizer is not None:
                feature = normalizer.forward_process(feature)
            speaker_id = value[:, -1].reshape(-1, 1)
            # print(speaker_id)
            # print(speaker_id.shape)
            test = np.concatenate((feature, speaker_id), axis=1)
            # print('2: ',test.shape)
            total_sp_speaker.append(test)
            # total_sp_speaker.append(speaker_id)
            # print(feature.shape)

    # print(total_sp)
    # print(total_speaker.shape)
    total_sp_speaker = np.concatenate(total_sp_speaker, axis=0)
    # print('3: ',total_sp_speaker.shape)
    # total_speaker = np.concatenate(total_speaker, axis=0)

    # if normalizer is not None:
    #    feature = normalizer.forward_process(feature)

    if data_format == 'NCHW':
        total_sp_speaker = total_sp_speaker.reshape([-1, 1, SP_DIM + 1, 1])

    elif data_format == 'NHWC':
        total_sp_speaker = total_sp_speaker.reshape([-1, SP_DIM + 1, 1, 1])

    else:
        pass

    # total_speaker = total_speaker.astype(np.int64)

    # print(total_sp_speaker)
    # return tf.train.shuffle_batch(
    #    [feature, speaker],
    #    batch_size,
    #    capacity=capacity,
    #    min_after_dequeue=min_after_dequeue,
    #    num_threads=num_threads,
    #    # enqueue_many=True,
    # )

    return total_sp_speaker


def read_whole_features(file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    total = []

    files = [file_name for file_name in glob.glob(join(file_pattern))]  # 1620(only including Testing Set)
    print('{} files found'.format(len(files)))
    for file_name in files:
        value = {}
        print('\rProcessing {}'.format(file_name), end='')
        with open(file_name, "rb") as reader:
            bytes_data = reader.read()
            data = np.fromstring(bytes_data, dtype=np.float32).reshape(-1, FEAT_DIM)
            value['sp'] = data[:, :SP_DIM]
            value['ap'] = data[:, SP_DIM: 2 * SP_DIM]
            value['f0'] = data[:, SP_DIM * 2]
            value['en'] = data[:, SP_DIM * 2 + 1]
            value['speaker'] = data[:, SP_DIM * 2 + 2].astype(np.int64)
            value['filename'] = file_name
        total.append(value)

    return total


def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    print(type(features['sp']))
    print(type(features['en']))
    en = np.reshape(features['en'], [-1, 1])
    sp = np.power(10., features['sp'])
    sp = en * sp
    if isinstance(features, dict):
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
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
    unseen = [5, 6]
    seen = [1, 2, 3, 4, 7, 8]
    features, labels, train = load_complete_feature_embeding()


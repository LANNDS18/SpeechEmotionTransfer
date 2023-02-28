import pandas as pd
import os
import librosa
import numpy as np


def read_RAVDESS_from_dir(data_path='.', num_emotion=8):
    data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename == '.DS_Store':
                continue
            file_path = os.path.join(dirname, filename)
            identifiers = filename.split('.')[0].split('-')
            emotion = (int(identifiers[2]))
            if emotion == num_emotion:
                emotion = 0
            if int(identifiers[3]) == 1:
                emotion_intensity = 'normal'
            else:
                emotion_intensity = 'strong'
            if int(identifiers[6]) % 2 == 0:
                gender = 'female'
            else:
                gender = 'male'

            new_data = pd.DataFrame.from_records([{"Emotion": emotion,
                                                   "Emotion intensity": emotion_intensity,
                                                   "Gender": gender,
                                                   "Path": file_path, }])

            data = pd.concat([data, new_data], ignore_index=True)

    return data


# load ESD data
def read_esd(esd_path='./ESD/'):
    EMOTIONS = {'Neutral': 1, 'Calm': 2, 'Happy': 3, 'Sad': 4, 'Angry': 5, 'Fear': 6, 'Disgust': 7, 'Surprise': 0}

    train_data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])
    test_data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])
    eva_data = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path'])

    res = [train_data, test_data, eva_data]

    for dir_actor in os.listdir(esd_path):
        if dir_actor == '.DS_Store' or dir_actor.endswith('.txt'):
            continue

        actor_path = os.path.join(esd_path, dir_actor)
        for emotion in os.listdir(actor_path):
            if emotion == '.DS_Store' or emotion.endswith('.txt'):  # is a txt file
                continue

            folders = ['evaluation', 'train', 'test']

            for folder in range(len(folders)):
                emo_path = os.path.join(actor_path, emotion, folders[folder])
                for wav_file in os.listdir(emo_path):
                    emotion_id = EMOTIONS[emotion]
                    wav_path = os.path.join(emo_path, wav_file)
                    intensity = 'normal'
                    if dir_actor in ['0001', '0002', '0003', '0007', '0009', '0015', '0016', '0017', '0018', '0019']:
                        gender = 'female'
                    else:
                        gender = 'male'
                    if not wav_file.endswith('.wav'):  # is a txt file
                        continue
                    data = res[folder]
                    new_data = pd.DataFrame.from_records([{"Emotion": emotion_id,
                                                           "Emotion intensity": intensity,
                                                           "Gender": gender,
                                                           "Path": wav_path, }])
                    res[folder] = pd.concat([data, new_data], ignore_index=True)
    return res


def load_signal(data, sample_rate=48000):
    SAMPLE_RATE = sample_rate
    signals = []
    max_len = 0
    for i, file_path in enumerate(data.Path):
        # only load 3 seconds
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
        signal = np.zeros((int(SAMPLE_RATE * 3, )))
        signal[:len(audio)] = audio
        signals.append(signal)
        print("\r Processed {}/{} files".format(i + 1, len(data)), end='')
    signals = np.stack(signals, axis=0)
    return signals


def split_RAVDESS_data(data, signals, emotions=None):
    if emotions is None:
        emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

    X = signals
    train_ind, test_ind, val_ind = [], [], []
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    for emotion in range(len(emotions)):
        emotion_ind = list(data.loc[data.Emotion == emotion, 'Emotion'].index)
        emotion_ind = np.random.permutation(emotion_ind)
        m = len(emotion_ind)
        ind_train = emotion_ind[:int(0.8 * m)]
        ind_val = emotion_ind[int(0.8 * m):int(0.9 * m)]
        ind_test = emotion_ind[int(0.9 * m):]
        X_train.append(X[ind_train, :])
        Y_train.append(np.array([emotion] * len(ind_train), dtype=np.int32))
        X_val.append(X[ind_val, :])
        Y_val.append(np.array([emotion] * len(ind_val), dtype=np.int32))
        X_test.append(X[ind_test, :])
        Y_test.append(np.array([emotion] * len(ind_test), dtype=np.int32))
        train_ind.append(ind_train)
        test_ind.append(ind_test)
        val_ind.append(ind_val)
    X_train = np.concatenate(X_train, 0)
    X_val = np.concatenate(X_val, 0)
    X_test = np.concatenate(X_test, 0)
    Y_train = np.concatenate(Y_train, 0)
    Y_val = np.concatenate(Y_val, 0)
    Y_test = np.concatenate(Y_test, 0)
    train_ind = np.concatenate(train_ind, 0)
    val_ind = np.concatenate(val_ind, 0)
    test_ind = np.concatenate(test_ind, 0)
    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
    print(f'X_val:{X_val.shape}, Y_val:{Y_val.shape}')
    print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')
    # check if all are unique
    unique, count = np.unique(np.concatenate([train_ind, test_ind, val_ind], 0), return_counts=True)
    print("Number of unique indexes is {}, out of {}".format(sum(count == 1), X.shape[0]))

    del X

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0 ** (num_bits - 1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    # Generate noisy signal
    return signal + K.T * noise


def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


if __name__ == '__main__':
    SAMPLE_RATE = 48000

    RAVDESS_data = read_RAVDESS_from_dir(data_path='./audio_speech_actors_01-24')

    signal = load_signal(RAVDESS_data, sample_rate=SAMPLE_RATE)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_RAVDESS_data(RAVDESS_data, signal,
                                                                        emotions={1: 'neutral', 2: 'calm', 3: 'happy',
                                                                                  4: 'sad', 5: 'angry', 6: 'fear',
                                                                                  7: 'disgust',
                                                                                  0: 'surprise'})

    # Data Argumentation
    aug_signals = []
    aug_labels = []
    for i in range(X_train.shape[0]):
        signal = X_train[i, :]
        augmented_signals = addAWGN(signal)
        for j in range(augmented_signals.shape[0]):
            aug_labels.append(RAVDESS_data.loc[i, "Emotion"])
            aug_signals.append(augmented_signals[j, :])

            new_data = RAVDESS_data.iloc[i]

            RAVDESS_data = pd.concat([RAVDESS_data, new_data])

        print("\r Processed {}/{} files".format(i + 1, X_train.shape[0]), end='')
    aug_signals = np.stack(aug_signals, axis=0)
    X_train = np.concatenate([X_train, aug_signals], axis=0)
    aug_labels = np.stack(aug_labels, axis=0)
    Y_train = np.concatenate([Y_train, aug_labels])
    print('')
    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')

    mel_train = []
    for i in range(X_train.shape[0]):
        mel_spectrogram = getMELspectrogram(X_train[i, :], sample_rate=SAMPLE_RATE)
        mel_train.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i + 1, X_train.shape[0]), end='')
    print('')
    mel_train = np.stack(mel_train, axis=0)
    del X_train
    X_train = mel_train
    print(mel_train.shape)

    mel_val = []
    print("Calculatin mel spectrograms for val set")
    for i in range(X_val.shape[0]):
        mel_spectrogram = getMELspectrogram(X_val[i, :], sample_rate=SAMPLE_RATE)
        mel_val.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i + 1, X_val.shape[0]), end='')
    print('')
    mel_val = np.stack(mel_val, axis=0)
    del X_val
    X_val = mel_val

    mel_test = []
    print("Calculatin mel spectrograms for test set")
    for i in range(X_test.shape[0]):
        mel_spectrogram = getMELspectrogram(X_test[i, :], sample_rate=SAMPLE_RATE)
        mel_test.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i + 1, X_test.shape[0]), end='')
    print('')
    mel_test = np.stack(mel_test, axis=0)
    del X_test
    X_test = mel_test

    print(f'X_train:{X_train.shape}, Y_train:{Y_train.shape}')
    print(f'X_val:{X_val.shape}, Y_val:{Y_val.shape}')
    print(f'X_test:{X_test.shape}, Y_test:{Y_test.shape}')

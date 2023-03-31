import torch
import numpy as np
from torchsummary import summary

from ParallelCNNTrans import ParallelModel
from preprocess_utils import read_RAVDESS_from_dir, load_signal, getMELspectrogram

from sklearn.preprocessing import StandardScaler

EMOTION = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
SAMPLE_RATE = 48000


def split_RAVDESS_data(data, signals, emotions=None):
    if emotions is None:
        emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

    X = []
    Y = []
    all_inds = []

    for emotion in range(len(emotions)):
        emotion_ind = list(data.loc[data.Emotion == emotion, 'Emotion'].index)
        emotion_ind = np.random.permutation(emotion_ind)
        ind = emotion_ind[:]

        X.append(signals[ind, :])
        Y.append(np.array([emotion] * len(ind), dtype=np.int32))

        all_inds.append(ind)

    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)

    print(f'X_train:{X.shape}, Y_train:{Y.shape}')
    # check if all are unique
    unique, count = np.unique(np.concatenate(all_inds, 0), return_counts=True)
    print("Number of unique indexes is {}, out of {}".format(sum(count == 1), X.shape[0]))

    return X, Y


def load_and_split_representation_dataset():
    RAVDESS_data = read_RAVDESS_from_dir('./audio_speech_actors_01-24')
    signals = load_signal(RAVDESS_data, SAMPLE_RATE)
    print(signals.shape)

    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

    X = []
    Y = []
    all_inds = []

    for emotion in range(len(emotions)):
        emotion_ind = list(RAVDESS_data.loc[RAVDESS_data.Emotion == emotion, 'Emotion'].index)
        emotion_ind = np.random.permutation(emotion_ind)
        ind = emotion_ind[:]

        X.append(signals[ind, :])
        Y.append(np.array([emotion] * len(ind), dtype=np.int32))

        all_inds.append(ind)

    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)

    print(f'X_train:{X.shape}, Y_train:{Y.shape}')
    # check if all are unique
    unique, count = np.unique(np.concatenate(all_inds, 0), return_counts=True)
    print("Number of unique indexes is {}, out of {}".format(sum(count == 1), X.shape[0]))

    mel_train = []
    for i in range(X.shape[0]):
        mel_spectrogram = getMELspectrogram(X[i, :], sample_rate=SAMPLE_RATE)
        mel_train.append(mel_spectrogram)
        print("\r Processed {}/{} files".format(i + 1, X.shape[0]), end='')
    print('')
    mel_train = np.stack(mel_train, axis=0)
    del X
    X = mel_train
    print(mel_train.shape)

    X = np.expand_dims(X, 1)

    scaler = StandardScaler()

    b, c, h, w = X.shape
    X = np.reshape(X, newshape=(b, -1))
    X = scaler.fit_transform(X)
    X = np.reshape(X, newshape=(b, c, h, w))

    return RAVDESS_data, X, Y


def generate_from_all():
    data, X, Y = load_and_split_representation_dataset()
    device = 'cpu' if torch.has_mps else 'cpu'
    model = ParallelModel(len(EMOTION)).to(device)
    model.load_state_dict(torch.load('./bin/models/cnn_transf_parallel_model.pt', map_location=device))
    summary(model)
    with torch.no_grad():
        X_tensor = torch.tensor(X, device=device).float()
        output_logits, output_softmax, complete_embedding = model(X_tensor)
    # save the embedding
    np.save('./bin/models/RAVDESS_complete_embedding222.npy', complete_embedding.cpu().numpy())

import os
def save_model():

    data, X, Y = load_and_split_representation_dataset()
    device = 'cpu' if torch.has_mps else 'cpu'
    model = ParallelModel(len(EMOTION)).to(device)
    model.load_state_dict(torch.load('./bin/models/cnn_transf_parallel_model.pt', map_location=device))
    summary(model)
    with torch.no_grad():
        X_tensor = torch.tensor(X, device=device).float()
        output_logits, output_softmax, complete_embedding = model(X_tensor)

    torch.save(model.state_dict(), './bin/models/cnn_transf_parallel_model.pt')
    print('Model is saved to {}'.format(os.path.join('./bin/models/cnn_transf_parallel_model.pt')))


def load_embedding():
    RAVDESS_data = read_RAVDESS_from_dir('./audio_speech_actors_01-24')
    complete_embedding = np.load('./bin/models/RAVDESS_complete_embedding.npy')
    return RAVDESS_data, complete_embedding


def get_emotion_representation(emotion_id, length=601, num_sample=5):
    RAVDESS_data, complete_embedding = load_embedding()
    emotion_index = RAVDESS_data[RAVDESS_data['Emotion'] == emotion_id].index
    print(len(emotion_index))
    rep = np.zeros((320,))
    count = 0
    res = np.ndarray(shape=(length, 320))
    for i in range(length):
        emb = np.random.choice(emotion_index)
        res[i] = complete_embedding[emb] * 10

    for i, d in enumerate(complete_embedding):
        if i in emotion_index:
            count += 1
            rep += complete_embedding[i]
        if count == num_sample:
            break
    rep /= count
    return rep


if __name__ == '__main__':
    save_model()
    # generate_from_all()
    """
    RAVDESS_data, complete_embedding = load_embedding()
    print(RAVDESS_data.shape)
    print(complete_embedding.shape)
    print(RAVDESS_data['Emotion'].value_counts())
    print(RAVDESS_data.head())
    get_emotion_representation(5)
    """

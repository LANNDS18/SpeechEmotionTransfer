import torch
import numpy as np
from torchsummary import summary

from ParallelCNNTrans import ParallelModel
from preprocess import read_RAVDESS_from_dir, load_signal, getMELspectrogram

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


def load_representation_data():
    RAVDESS_data = read_RAVDESS_from_dir('./audio_speech_actors_01-24')
    signals = load_signal(RAVDESS_data, SAMPLE_RATE)
    print(signals.shape)
    X, Y = split_RAVDESS_data(RAVDESS_data, signals, EMOTION)

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
    data, X, Y = load_representation_data()
    device = 'cpu' if torch.has_mps else 'cpu'
    model = ParallelModel(len(EMOTION)).to(device)
    model.load_state_dict(torch.load('./bin/models/cnn_transf_parallel_model.pt', map_location=device))
    summary(model)
    with torch.no_grad():
        X_tensor = torch.tensor(X, device=device).float()
        output_logits, output_softmax, complete_embedding = model(X_tensor)
    # save the embedding
    np.save('./bin/models/RAVDESS_complete_embedding.npy', complete_embedding.cpu().numpy())


def load_embedding():
    RAVDESS_data = read_RAVDESS_from_dir('./audio_speech_actors_01-24')
    complete_embedding = np.load('./bin/models/RAVDESS_complete_embedding.npy')
    return RAVDESS_data, complete_embedding


if __name__ == '__main__':
    RAVDESS_data, complete_embedding = load_embedding()
    print(RAVDESS_data.shape)
    print(complete_embedding.shape)
    print(RAVDESS_data['Emotion'].value_counts())

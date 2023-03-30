import os
from pathlib import Path
from statistics import mean

import librosa
import pandas as pd
from mel_cepstral_distance import get_metrics_mels

from analyzer import load_test_data


# get_metrics_mels()

# get_metrics_wavs()

def compute_MCD(src: int, tgt_emo: int, synth_folder: str):
    test_data = load_test_data(src, tgt_emo)  ## list of dict
    syn_files = os.listdir(synth_folder)  ## list of synth file names
    assert len(test_data) == len(syn_files)

    # tgt_files = []
    # for test in len(test_data):
    #     tgt_files.append(test['Target_Path'])

    filename_list = []
    mcd_list = []
    penalty_list = []
    mcd_with_penalty_list = []

    for i in range(len(test_data)):

        filename2 = Path(test_data[i]['Target_Path'])
        if not filename2.is_file():
            print("Target file not exsit: ", filename2)
            continue

        wavID = os.path.basename(test_data[i]['Path'])
        if wavID not in syn_files:
            print(wavID, " not in synth_folder")
            continue
        filename1 = Path(os.path.join(synth_folder, wavID))
        print("file1", filename1)

        print("Processing -----------{}".format(wavID))

        print("synth audio: ", filename1)
        print("Target audio: ", filename2)

        ## method 1
        # mcd, penalty,_ = get_metrics_wavs(filename1, filename2) ## not work

        ## method 2
        audio_1, sample_rate_1 = librosa.load(filename1, duration=3, sr=16000)
        m1 = librosa.feature.melspectrogram(y=audio_1,
                                            sr=sample_rate_1,
                                            n_fft=1024,
                                            )

        audio_2, sample_rate_2 = librosa.load(filename2, duration=3, offset=0.5, sr=16000)
        m2 = librosa.feature.melspectrogram(y=audio_2,
                                            sr=sample_rate_2,
                                            n_fft=1024,
                                            )

        import numpy as np

        coefficient = 10e-40
        m1 = np.where(m1 == 0, coefficient, m1)
        m2 = np.where(m2 == 0, coefficient, m2)

        mcd, penalty, _ = get_metrics_mels(m1, m2)

        mcd_list.append(mcd)
        penalty_list.append(penalty)
        mcd_with_penalty_list.append(mcd + penalty)
        filename_list.append(wavID)

        print("finish {} file--------------".format(i))

    filename_list.append("Average")

    avg_MCD_with_Penalty = mean(mcd_with_penalty_list)
    mcd_with_penalty_list.append(avg_MCD_with_Penalty)

    avg_MCD = mean(mcd_list)
    mcd_list.append(avg_MCD)

    avg_penalty = mean(penalty_list)
    penalty_list.append(avg_penalty)

    MCD = pd.DataFrame(list(zip(filename_list, mcd_with_penalty_list, mcd_list, penalty_list)),
                       columns=['File', 'MCD with Penalty', 'MCD', 'Penalty'])

    print(MCD.tail())
    return MCD


if __name__ == '__main__':
    compute_MCD(1, 5, "./converted/model_VAE_1/1_to_5")

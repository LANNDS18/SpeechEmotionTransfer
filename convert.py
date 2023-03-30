import os
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from analyzer import Tanhize, pw2wav, load_test_data, SP_DIM
from emotion_representation import get_emotion_representation
from trainer import DEVICE
from VAE_Trainer import VAE_Trainer
from VAW_Trainer import VAW_Trainer


def nh_to_nchw(x):
    return x.reshape(-1, 1, 513, 1)


def convert_f0(f0, src, trg):
    # print(f0)

    # mu and std of log f0 of traininf audio of the speaker
    mu_s, std_s = np.fromfile(os.path.join('./bin/etc', '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join('./bin/etc', '{}.npf'.format(trg)), np.float32)

    # if f0>1, element replaced by log(f0)
    # lf0 = np.where(f0 > 1., np.log(f0), f0)
    lf0 = np.where(f0 > 1., np.log(f0, where=f0 > 1), f0)
    # print(np.all(lf0==lf0_2))

    # if lf0>1, standardaize the element
    lf0 = np.where(lf0 > 1., (lf0 - mu_s) / std_s * std_t + mu_t, lf0)

    # if lf0 >1, exp(element)
    lf0 = np.where(lf0 > 1., np.exp(lf0), lf0)
    return lf0


def get_default_output(logdir_root):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, 'output', STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))
    return logdir


def make_output_wav_name(output_dir, filename):
    # print(filename)
    # print(type(filename))
    basename = filename
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    # print('Processing {}'.format(basename))

    # return os.path.join(
    #     output_dir,
    #     '{}-{}-{}.wav'.format(args.src, args.trg, basename)
    # )
    return os.path.join(output_dir, "{}.wav".format(basename))


def convert(model_path: str, src: int, tgt: int):
    FS = 16000

    # load model
    model = torch.load(model_path)
    # model = joblib.load('./model/'+str(args.model_name)+'.pt')

    corpus_name = 'vcc2016'

    normalizer = Tanhize(
        xmax=np.fromfile('./bin/etc/{}_xmax.npf'.format(corpus_name)),
        xmin=np.fromfile('./bin/etc/{}_xmin.npf'.format(corpus_name)),
    )

    # test set src
    # extract a list of feature dict, each dict corresponds one audio file, dict contains  `sp`, `ap`, `f0`, `filename`
    total_features = load_test_data(src, tgt)
    # print(len(total_features))

    for features in total_features:  # the features in one test file

        sp = features['Feature'][:, :SP_DIM]  # (601, 513)
        # print(features['Feature'].shape)
        # print(sp.shape)
        # print(type(sp))

        for index, frame in enumerate(sp):
            sp[index] = normalizer.forward_process(frame)

        x = sp
        x = nh_to_nchw(x)  # reshape  ## (601, 1, 513, 1)
        # print(x.shape)

        x = torch.FloatTensor(x).to(device=DEVICE)

        # source f0
        f0 = features['Feature'][:, SP_DIM]
        # convert f0 s2t ## not sure why do this
        f0 = convert_f0(f0, src, tgt)  # ? log + norm ## convert from f0_s
        f0 = torch.FloatTensor(f0).view(-1, 1).to(device=DEVICE)  # [601,1]
        # print(f0.shape)

        # target embedding
        emb = get_emotion_representation(tgt, x.shape[0]).reshape(1, -1)  # [1, 320]
        # print('emb shape:', emb.shape)
        emb = np.repeat(emb, x.shape[0], 0)
        # print(emb.shape)
        emb = torch.FloatTensor(emb).to(device=DEVICE)  # [601, 320]

        # model inference
        z, _ = model.Encoder(x)
        # emb = torch.zeros(emb.shape).to(device=DEVICE)
        concat = torch.cat((z, f0, emb), 1)  # [601, 128+1+320=449]
        x_t, _ = model.G(concat)  # NOTE: the API yields NHWC format

        x_t = x_t.cpu()
        x_t = torch.squeeze(x_t)  # new generated sp ## [601, 513]

        x_t = x_t.data.numpy()
        for index, sp in enumerate(x_t):
            x_t[index] = normalizer.backward_process(sp)  # inverse norm: new sp s2t (np array)
        print('backward_process.finish')
        # print(type(x_t))

        # print(x_t)
        # print(emb)

        reconst = dict()
        reconst['sp'] = x_t
        reconst['f0'] = features['Feature'][:, SP_DIM]
        reconst['ap'] = features['ap_en'][:, :SP_DIM]
        reconst['en'] = features['ap_en'][:, SP_DIM]
        print('=-=-=-=-=-=')
        y = pw2wav(reconst)

        model_name = Path(model_path).stem
        output_dir = "./converted/" + model_name + "/" + str(src) + "_to_" + str(tgt) + "/"
        # print(output_dir)
        oFilename = make_output_wav_name(output_dir, features['Path'])
        # print(oFilename)

        print('\rProcessing {}'.format(oFilename), end='')

        if not os.path.exists(os.path.dirname(oFilename)):
            try:
                os.makedirs(os.path.dirname(oFilename))
            except OSError as exc:  # Guard against race condition
                print('error')
                pass

        sf.write(oFilename, y, FS)

    print('==finish==')


if __name__ == '__main__':
    convert("./model/model_VAW_00.pt", 2, 5)

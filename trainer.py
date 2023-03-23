import os
from math import pi

import torch.optim as optim
from torch.utils.data import DataLoader

from VAW_GAN import *

EPSILON = torch.tensor([1e-6], requires_grad=False)  # .cuda after tensor
PI = torch.tensor([pi], requires_grad=False)  # .cuda after tensor

LR = 1e-4
EPOCH_VAE = 5
EPOCH_VAWGAN = 11

FEATURE_DIM = 513 + 1 + 320 + 1
SP_DIM = 513
F0_DIM = 1
EMBED_DIM = 320
NUM_SAMPLE = 10


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Trainer:

    def __init__(self):

        self.G = G()  # .cuda()
        self.G.apply(weights_init)
        self.D = D()  # .cuda()
        self.D.apply(weights_init)
        self.Encoder = Encoder()  # .cuda()
        self.Encoder.apply(weights_init)
        self.batch_size = 256  # batch size
        self.source = None
        self.target = None

    def load_data(self, x, y):
        self.source = x
        self.target = y

    def circuit_loop(self, feature, label, f0, emb):

        z_mu, z_lv = self.Encoder(feature)
        z = GaussianSampleLayer(z_mu, z_lv)
        x_logit, x_feature = self.D(feature)

        concat = torch.cat((z, f0, emb), 1)

        xh, xh_sig_logit = self.G(concat)  # [256,128] #[256,1]
        xh_logit, xh_feature = self.D(xh)  # xh_logit[256,1]

        """
        print("feature_shape: ", feature.shape)
        print("f0_shape: ", f0.shape)
        print("emb_shape: ", emb.shape)
        print("x_feature: ", x_feature.shape)
        print("z_shape: ", z.shape)
        """

        return dict(
            z=z,
            z_mu=z_mu,
            z_lv=z_lv,
            xh=xh,
            xh_sig_logit=xh_sig_logit,
            x_logit=x_logit,
            x_feature=x_feature,
            xh_logit=xh_logit,
            xh_feature=xh_feature,

        )

    def train(self, device='cpu'):

        gan_loss = 50000
        x_feature = torch.FloatTensor(self.batch_size, 1, 513, 1)  # .cuda()  # NHWC
        x_label = torch.FloatTensor(self.batch_size)  # .cuda()
        y_feature = torch.FloatTensor(self.batch_size, 1, 513, 1)  # .cuda()  # NHWC
        y_label = torch.FloatTensor(self.batch_size)  # .cuda()

        x_f0 = torch.FloatTensor(self.batch_size, F0_DIM)  # .cuda()  # NHWC
        x_emb = torch.FloatTensor(self.batch_size, EMBED_DIM)  # .cuda()  # NHWC
        y_f0 = torch.FloatTensor(self.batch_size, F0_DIM)  # .cuda()  # NHWC
        y_emb = torch.FloatTensor(self.batch_size, EMBED_DIM)  # .cuda()  # NHWC

        """
        x_feature = torch.tensor(x_feature)
        x_label = torch.tensor(x_label, requires_grad=False)
        y_feature = torch.tensor(y_feature)
        y_label = torch.tensor(y_label, requires_grad=False)

        x_f0 = torch.tensor(x_f0, requires_grad=False)
        x_emb = torch.tensor(x_emb, requires_grad=False)
        y_f0 = torch.tensor(y_f0, requires_grad=False)
        y_emb = torch.tensor(y_emb, requires_grad=False)
        """

        optimD = optim.RMSprop([{'params': self.D.parameters()}], lr=LR)
        optimG = optim.RMSprop([{'params': self.G.parameters()}], lr=LR)
        optimE = optim.RMSprop([{'params': self.Encoder.parameters()}], lr=LR)

        schedulerD = torch.optim.lr_scheduler.StepLR(optimD, step_size=10, gamma=0.1)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimG, step_size=10, gamma=0.1)
        schedulerE = torch.optim.lr_scheduler.StepLR(optimE, step_size=10, gamma=0.1)

        Data = DataLoader(
            ConcatDataset(self.source, self.target),
            batch_size=self.batch_size, shuffle=True, num_workers=1)

        # print('N H W C')

        for epoch in range(EPOCH_VAE):

            # schedulerD.step()
            # schedulerG.step()
            # schedulerE.step()
            for index, (s_data, t_data) in enumerate(Data):
                # Source
                feature_1 = s_data[:, :513, :, :].permute(0, 3, 1, 2)  # NHWC ==> NCHW
                f0_1 = s_data[:, SP_DIM, :, :].view(-1, 1)
                embed_1 = s_data[:, SP_DIM + F0_DIM: SP_DIM + F0_DIM + EMBED_DIM, :, :].permute(0, 3, 1, 2)
                embed_1 = embed_1.view(-1, EMBED_DIM)
                label_1 = s_data[:, -1, :, :].view(len(s_data))

                if len(s_data) != self.batch_size:
                    continue # todo: fix last batch not full

                x_feature.data.resize_(feature_1.size())
                x_label.data.resize_(len(s_data))

                x_feature.data.copy_(feature_1)
                x_label.data.copy_(label_1)

                x_f0.data.resize_(f0_1.size())
                x_f0.data.copy_(f0_1)

                x_emb.data.resize_(embed_1.size())
                x_emb.data.copy_(embed_1)

                # Target
                feature_2 = t_data[:, :513, :, :].permute(0, 3, 1, 2)  # NHWC ==> NCHW
                label_2 = t_data[:, -1, :, :].view(len(t_data))
                f0_2 = t_data[:, SP_DIM, :, :].view(-1, 1)
                embed_2 = t_data[:, SP_DIM + F0_DIM: SP_DIM + F0_DIM + EMBED_DIM, :, :].permute(0, 3, 1, 2)
                embed_2 = embed_2.view(-1, EMBED_DIM)

                y_feature.data.resize_(feature_2.size())
                y_label.data.resize_(len(t_data))

                y_feature.data.copy_(feature_2)
                y_label.data.copy_(label_2)

                y_f0.data.resize_(f0_2.size())
                y_f0.data.copy_(f0_2)

                y_emb.data.resize_(embed_2.size())
                y_emb.data.copy_(embed_2)

                s = self.circuit_loop(x_feature, label_1, x_f0, x_emb)
                t = self.circuit_loop(y_feature, y_label, y_f0, y_emb)
                # Source 2 Target
                s2t = self.circuit_loop(x_feature, y_label, x_f0, y_emb)

                loss = dict()
                loss['conv_s2t'] = reconst_loss(t['x_logit'], s2t['xh_logit'])
                loss['conv_s2t'] *= 100

                loss['KL(z)'] = torch.mean(
                        GaussianKLD(
                            s['z_mu'], s['z_lv'],
                            torch.zeros_like(s['z_mu']), torch.zeros_like(s['z_lv']))) + \
                    torch.mean(
                        GaussianKLD(
                            t['z_mu'], t['z_lv'],
                            torch.zeros_like(t['z_mu']), torch.zeros_like(t['z_lv'])))
                loss['KL(z)'] /= 2.0

                loss['Dis'] = torch.mean(
                        GaussianLogDensity(
                            x_feature.view(-1, 513),
                            s['xh'].view(-1, 513),
                            torch.zeros_like(x_feature.view(-1, 513)))) + \
                    torch.mean(
                        GaussianLogDensity(
                            y_feature.view(-1, 513),
                            t['xh'].view(-1, 513),
                            torch.zeros_like(y_feature.view(-1, 513))))
                loss['Dis'] /= - 2.0

                optimE.zero_grad()
                obj_Ez = loss['KL(z)'] + loss['Dis']
                obj_Ez.backward(retain_graph=True)

                optimG.zero_grad()
                obj_Gx = loss['Dis']
                obj_Gx.backward()

                optimE.step()
                optimG.step()

                print("Epoch:[%d|%d]\tIteration:[%d|%d]\tW: %.3f\tKL(Z): %.3f\tDis: %.3f" % (
                    epoch + 1, EPOCH_VAWGAN + EPOCH_VAE, index + 1, len(Data),
                    loss['conv_s2t'], loss['KL(z)'], loss['Dis']))

        exit()

        for epoch in range(EPOCH_VAWGAN):

            schedulerD.step()
            schedulerG.step()
            schedulerE.step()
            for index, (s_data, t_data) in enumerate(Data):

                # Source
                feature_1 = s_data[:, :513, :, :].permute(0, 3, 1, 2)  # NHWC ==> NCHW
                label_1 = s_data[:, -1, :, :].view(len(s_data))

                x_feature.data.resize_(feature_1.size())
                x_label.data.resize_(len(s_data))

                x_feature.data.copy_(feature_1)
                x_label.data.copy_(label_1)

                # Target
                feature_2 = t_data[:, :513, :, :].permute(0, 3, 1, 2)  # NHWC ==> NCHW
                label_2 = t_data[:, -1, :, :].view(len(t_data))

                y_feature.data.resize_(feature_2.size())
                y_label.data.resize_(len(t_data))

                y_feature.data.copy_(feature_2)
                y_label.data.copy_(label_2)

                t = dict()

                # Source 2 Target
                s2t = dict()

                loss = dict()

                if (epoch + EPOCH_VAE == EPOCH_VAE and index < 25) or (index % 100 == 0):
                    D_Iter = 100
                else:
                    D_Iter = 10

                for D_index in range(D_Iter):
                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    # Target result
                    optimD.zero_grad()
                    t = self.circuit_loop(y_feature, y_label)
                    # Source 2 Target result
                    s2t = self.circuit_loop(x_feature, y_label)

                    loss['conv_s2t'] = \
                        reconst_loss(t['x_logit'], s2t['xh_logit'])

                    loss['conv_s2t'] *= 100
                    # print ("%.3f\t" %(loss['conv_s2t']))
                    # print(   loss )
                    obj_Dx = -0.01 * loss['conv_s2t']
                    obj_Dx.backward(retain_graph=True)
                    optimD.step()

                    # Source result
                s = self.circuit_loop(x_feature, x_label)

                loss['KL(z)'] = \
                    torch.mean(
                        GaussianKLD(
                            s['z_mu'], s['z_lv'],
                            torch.zeros_like(s['z_mu']), torch.zeros_like(s['z_lv']))) + \
                    torch.mean(
                        GaussianKLD(
                            t['z_mu'], t['z_lv'],
                            torch.zeros_like(t['z_mu']), torch.zeros_like(t['z_lv'])))
                loss['KL(z)'] /= 2.0

                loss['Dis'] = \
                    torch.mean(
                        GaussianLogDensity(
                            x_feature.view(-1, 513),
                            s['xh'].view(-1, 513),
                            torch.zeros_like(x_feature.view(-1, 513)))) + \
                    torch.mean(
                        GaussianLogDensity(
                            y_feature.view(-1, 513),
                            t['xh'].view(-1, 513),
                            torch.zeros_like(y_feature.view(-1, 513))))
                loss['Dis'] /= - 2.0
                # print(   loss )

                # print ("%.3f\t" %(loss['conv_s2t']))

                optimE.zero_grad()
                obj_Ez = loss['KL(z)'] + loss['Dis']
                obj_Ez.backward(retain_graph=True)
                optimE.step()

                optimG.zero_grad()
                obj_Gx = loss['Dis'] + 50 * loss['conv_s2t']
                obj_Gx.backward()
                optimG.step()
                print(
                    "Epoch:[%d|%d]\tIteration:[%d|%d]\t[D_loss: %.3f\tG_loss: %.3f\tE_loss: %.3f]\t[S2T: %.3f\tKL(z): "
                    "%.3f\tDis: %.3f]" % (
                        EPOCH_VAE + epoch + 1, EPOCH_VAWGAN + EPOCH_VAE, index + 1, len(Data),
                        -0.01 * loss['conv_s2t'], loss['Dis'] + 50 * loss['conv_s2t'], loss['Dis'] + loss['KL(z)'],
                        loss['conv_s2t'], loss['KL(z)'], loss['Dis']))

                if epoch == EPOCH_VAWGAN - 1 and index == (len(Data) - 2):
                    print('================= store model ==================')
                    filename = './model/model_' + str(epoch + EPOCH_VAE + 1) + '.pt'
                    if not os.path.exists(os.path.dirname(filename)):
                        try:
                            os.makedirs(os.path.dirname(filename))
                        except OSError as exc:  # Guard against race condition
                            print('error')
                            pass

                    torch.save(self, filename)
                    print('=================Finish store model ==================')
                    gan_loss = obj_Gx


def reconst_loss(x, xh):
    return torch.mean(x) - torch.mean(xh)


def GaussianSampleLayer(z_mu, z_lv):
    std = torch.sqrt(torch.exp(z_lv))
    eps = torch.randn_like(std)
    return eps.mul(std).add_(z_mu)


def GaussianLogDensity(x, mu, log_var):
    c = torch.log(2. * PI)
    var = torch.exp(log_var)
    x_mu2 = torch.mul(x - mu, x - mu)  # [Issue] not sure the dim works or not?
    x_mu2_over_var = torch.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = torch.sum(log_prob, 1)  # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''

    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    mu_diff_sq = torch.mul(mu1 - mu2, mu1 - mu2)
    dimwise_kld = .5 * (
            (lv2 - lv1) + torch.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)

    return torch.sum(dimwise_kld, 1)

import os
import pickle
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from math import pi
from VAW_GAN import Encoder, D, G, weights_init
from VAE_Trainer import VAE_Trainer  # must be import

from util import GaussianKLD, GaussianLogDensity, GaussianSampleLayer, reconst_loss, ConcatDataset,validate_log_dirs

LR = 1e-4

FEATURE_DIM = 513 + 1 + 320 + 1
SP_DIM = 513
F0_DIM = 1
EMBED_DIM = 320
NUM_SAMPLE = 10

DEVICE = torch.device('mps') if torch.has_mps else torch.device('cpu')

EPSILON = torch.tensor([1e-6], requires_grad=False).to(DEVICE)
PI = torch.tensor([pi], requires_grad=False).to(DEVICE)


class VAW_Trainer:

    def __init__(self, name, vae_load_dir):

        self.G = G().to(device=DEVICE)
        self.G.apply(weights_init)
        self.D = D().to(device=DEVICE)
        self.D.apply(weights_init)
        self.Encoder = Encoder().to(device=DEVICE)
        self.Encoder.apply(weights_init)
        self.batch_size = 256
        self.source = None
        self.target = None
        self.name = name
        torch.autograd.set_detect_anomaly(True)
        self.VAE_load_dir = vae_load_dir
        if not vae_load_dir:
            print("To train pure VAW-GAN, the pre-trained VAE dir must be specified")
            exit(999)
        self.dirs = validate_log_dirs(self.name)['logdir']
        os.makedirs(self.dirs)

    def load_data(self, x, y):
        self.source = x
        self.target = y

    def circuit_loop(self, feature, f0, emb):

        z_mu, z_lv = self.Encoder(feature)
        z = GaussianSampleLayer(z_mu, z_lv)
        x_logit, x_feature = self.D(feature)

        concat = torch.cat((z, f0, emb), 1)

        xh, xh_sig_logit = self.G(concat)
        xh_logit, xh_feature = self.D(xh)

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

    def train(self, num_epoch=20):

        print(f"Num of Epochs = {num_epoch}")

        model_type = VAE_Trainer
        model = torch.load(self.VAE_load_dir, map_location=DEVICE)
        self.D = model.D
        self.G = model.G

        gan_loss = 50000
        x_feature = torch.FloatTensor(self.batch_size, 1, FEATURE_DIM, 1).to(device=DEVICE)
        x_label = torch.FloatTensor(self.batch_size).to(device=DEVICE)
        y_feature = torch.FloatTensor(self.batch_size, 1, FEATURE_DIM, 1).to(device=DEVICE)
        y_label = torch.FloatTensor(self.batch_size).to(device=DEVICE)

        x_f0 = torch.FloatTensor(self.batch_size, F0_DIM).to(device=DEVICE)
        x_emb = torch.FloatTensor(self.batch_size, EMBED_DIM).to(device=DEVICE)
        y_f0 = torch.FloatTensor(self.batch_size, F0_DIM).to(device=DEVICE)
        y_emb = torch.FloatTensor(self.batch_size, EMBED_DIM).to(device=DEVICE)

        optimD = optim.RMSprop([{'params': self.D.parameters()}], lr=LR)
        optimG = optim.RMSprop([{'params': self.G.parameters()}], lr=LR)
        optimE = optim.RMSprop([{'params': self.Encoder.parameters()}], lr=LR)

        schedulerD = torch.optim.lr_scheduler.StepLR(optimD, step_size=10, gamma=0.1)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimG, step_size=10, gamma=0.1)
        schedulerE = torch.optim.lr_scheduler.StepLR(optimE, step_size=10, gamma=0.1)

        Data = DataLoader(
            ConcatDataset(self.source, self.target),
            batch_size=self.batch_size, shuffle=True, num_workers=1)

        for epoch in range(num_epoch):

            # initialize empty lists to store the losses
            conv_s2t_loss = []
            KL_z_loss = []
            Dis_loss = []
            d_loss = []
            g_loss = []
            e_loss = []

            for index, (s_data, t_data) in enumerate(Data):

                # Source
                feature_1 = s_data[:, :513, :, :].permute(0, 3, 1, 2)
                label_1 = s_data[:, -1, :, :].view(len(s_data))
                f0_1 = s_data[:, SP_DIM, :, :].view(-1, 1)
                embed_1 = s_data[:, SP_DIM + F0_DIM:SP_DIM + F0_DIM + EMBED_DIM, :, :].permute(0, 3, 1, 2).view(-1,
                                                                                                                EMBED_DIM)
                x_feature.resize_(feature_1.size())
                x_label.resize_(len(s_data))
                x_f0.resize_(f0_1.size())
                x_emb.resize_(embed_1.size())

                x_feature.copy_(feature_1)
                x_label.copy_(label_1)
                x_f0.copy_(f0_1)
                x_emb.copy_(embed_1)

                # Target
                feature_2 = t_data[:, :513, :, :].permute(0, 3, 1, 2)
                label_2 = t_data[:, -1, :, :].view(len(t_data))
                f0_2 = t_data[:, SP_DIM, :, :].view(-1, 1)
                embed_2 = t_data[:, SP_DIM + F0_DIM: SP_DIM + F0_DIM + EMBED_DIM, :, :] \
                    .permute(0, 3, 1, 2).view(-1, EMBED_DIM)

                y_feature.resize_(feature_2.size())
                y_label.resize_(len(t_data))
                y_f0.resize_(f0_2.size())
                y_emb.resize_(embed_2.size())

                y_feature.copy_(feature_2)
                y_label.copy_(label_2)
                y_f0.copy_(f0_2)
                y_emb.copy_(embed_2)

                t = dict()

                # Source 2 Target
                s2t = dict()

                loss = dict()

                if (epoch == 0 and index < 25) or (index % 100 == 0):
                    D_Iter = 100
                else:
                    D_Iter = 10

                for D_index in range(D_Iter):
                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    # Target result
                    optimD.zero_grad()
                    t = self.circuit_loop(y_feature, y_f0, y_emb)
                    # Source 2 Target result
                    s2t = self.circuit_loop(x_feature, x_f0, y_emb)

                    loss['conv_s2t'] = reconst_loss(t['x_logit'], s2t['xh_logit'])
                    loss['conv_s2t'] *= 100

                    # print("%.3f\t" % (loss['conv_s2t']))
                    # print(loss)

                    if D_index != D_Iter - 1:
                        # if not last epoch, run normally
                        obj_Dx = -0.01 * loss['conv_s2t']
                        obj_Dx.backward(retain_graph=True)
                        optimD.step()
                    else:
                        break

                # target result
                # t = self.circuit_loop(y_feature, y_f0, y_emb)
                # Source result
                s = self.circuit_loop(x_feature, x_f0, x_emb)

                loss['KL(z)'] = torch.mean(
                    GaussianKLD(
                        s['z_mu'], s['z_lv'],
                        torch.zeros_like(s['z_mu']), torch.zeros_like(s['z_lv']), EPSILON)
                ) + torch.mean(
                    GaussianKLD(
                        t['z_mu'], t['z_lv'],
                        torch.zeros_like(t['z_mu']), torch.zeros_like(t['z_lv']), EPSILON)
                )
                loss['KL(z)'] /= 2.0

                loss['Dis'] = torch.mean(
                    GaussianLogDensity(
                        x_feature.view(-1, 513),
                        s['xh'].view(-1, 513),
                        torch.zeros_like(x_feature.view(-1, 513)), PI, EPSILON)
                ) + torch.mean(
                    GaussianLogDensity(
                        y_feature.view(-1, 513),
                        t['xh'].view(-1, 513),
                        torch.zeros_like(y_feature.view(-1, 513)), PI, EPSILON)
                )

                loss['Dis'] /= - 2.0
                # print(loss)
                # print("%.3f\t" % (loss['conv_s2t']))

                optimE.zero_grad()
                obj_Ez = loss['KL(z)'] + loss['Dis']
                obj_Ez.backward(retain_graph=True)

                optimG.zero_grad()
                obj_Gx = loss['Dis'] + 50 * loss['conv_s2t']
                obj_Gx.backward(retain_graph=True)

                optimD.zero_grad()
                # if last epoch, update G as well,
                obj_Dx = -0.01 * loss['conv_s2t']
                obj_Dx.backward()

                optimE.step()
                optimG.step()
                optimD.step()

                conv_s2t_loss.append([loss['conv_s2t']])
                KL_z_loss.append([loss['KL(z)']])
                Dis_loss.append([loss['Dis']])
                d_loss.append([-0.01 * loss['conv_s2t']])
                g_loss.append([loss['Dis'] + 50 * loss['conv_s2t']])
                e_loss.append(loss['Dis'] + loss['KL(z)'])

                print(
                    "Epoch:[%d|%d]\tIteration:[%d|%d]\t[D_loss: %.3f\tG_loss: %.3f\tE_loss: %.3f]\t[S2T: %.3f\tKL(z): "
                    "%.3f\tDis: %.3f]" % (epoch + 1, num_epoch, index + 1, len(Data),
                                          -0.01 * loss['conv_s2t'], loss['Dis'] + 50 * loss['conv_s2t'],
                                          loss['Dis'] + loss['KL(z)'],
                                          loss['conv_s2t'], loss['KL(z)'], loss['Dis']))

                if epoch == num_epoch - 1 and index == (len(Data) - 2):
                    print('================= store model ==================')
                    filename = f'./model/model_{self.name}.pt'
                    if not os.path.exists(os.path.dirname(filename)):
                        try:
                            os.makedirs(os.path.dirname(filename))
                        except OSError:  # Guard against race condition
                            print('error')
                            pass

                    torch.save(self, filename)
                    print('=================Finish store model ==================')
                    gan_loss = obj_Gx

            # save the three loss lists to a local storage using pickle
            with open(f'./{self.dirs}/model_{self.name}_epoch{epoch}.pkl', 'wb') as f:
                pickle.dump((conv_s2t_loss, KL_z_loss, Dis_loss, d_loss, g_loss, e_loss), f)

            schedulerD.step()  # should be called after step()
            schedulerG.step()  # should be called after step()
            schedulerE.step()  # should be called after step()


if __name__ == '__main__':
    from analyzer import divide_into_source_target

    source = [1]
    target = 5
    epoch = 15

    source_data, target_data = divide_into_source_target(source, target)
    machine = VAW_Trainer(name='VAW_VAE_1_to_5', vae_load_dir='./model/model_VAE_1_to_5.pt')
    machine.load_data(source_data, target_data)
    machine.train(num_epoch=epoch)

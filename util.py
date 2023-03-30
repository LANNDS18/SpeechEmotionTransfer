import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset


def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def get_default_logdir(logdir_root):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))
    return logdir


def validate_log_dirs(name=None):
    """ Create a default log dir (if necessary) """
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir_root = 'logdir'
    logdir = os.path.join(logdir_root, name+'_'+STARTED_DATESTRING)
    restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from,
    }


def melcd(array1, array2):
    """Calculate mel-cepstrum distortion
    Calculate mel-cepstrum distortion between the arrays.
    This function assumes the shapes of arrays are same.
    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`) or shape (`dim`)
        Arrays of original and target.
    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion
    """
    if array1.shape != array2.shape:
        raise ValueError(
            "The shapes of both arrays are different \
            : {} / {}".format(array1.shape, array2.shape))

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) \
              * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise ValueError("Dimension mismatch")

    return mcd


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def reconst_loss(x, xh):
    return torch.mean(x) - torch.mean(xh)


def GaussianSampleLayer(z_mu, z_lv):
    std = torch.sqrt(torch.exp(z_lv))
    eps = torch.randn_like(std)
    return eps.mul(std).add_(z_mu)


def GaussianLogDensity(x, mu, log_var, PI, EPSILON):
    c = torch.log(2. * PI)
    var = torch.exp(log_var)
    x_mu2 = torch.mul(x - mu, x - mu)  # [Issue] not sure the dim works or not?
    x_mu2_over_var = torch.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = torch.sum(log_prob, 1)  # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2, EPSILON):
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

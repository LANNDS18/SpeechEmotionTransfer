import os
import sys
from datetime import datetime


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


def validate_log_dirs():
    """ Create a default log dir (if necessary) """

    logdir_root = 'logdir'

    logdir = get_default_logdir(logdir_root)

    restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from,
    }

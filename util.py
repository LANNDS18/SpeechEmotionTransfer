import json
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


def validate_log_dirs(args):
    ''' Create a default log dir (if necessary) '''

    def get_default_logdir(logdir_root):
        STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
        logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
        print('Using default logdir: {}'.format(logdir))
        return logdir

    if args.logdir and args.restore_from:
        raise ValueError(
            'You can only specify one of the following: ' +
            '--logdir and --restore_from')

    if args.logdir and args.log_root:
        raise ValueError(
            'You can only specify either --logdir or --logdir_root')

    if args.logdir_root is None:
        logdir_root = 'logdir'

    if args.logdir is None:
        logdir = get_default_logdir(logdir_root)

    # Note: `logdir` and `restore_from` are exclusive
    if args.restore_from is None:
        restore_from = logdir
    else:
        restore_from = args.restore_from

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from,
    }

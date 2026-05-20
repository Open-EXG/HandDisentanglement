from argparse import ArgumentParser
from copy import deepcopy
import glob
import numpy as np
import os
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

from settings import Config
from data_module import DataModule, DataModule_session
from models import *

## ！！pl版本要求！！conda install pytorch-lightning=1.9.4 -c conda-forge


def main(args):
    if args.mode=='getpath':
        print(args.output_log)
        return None

    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.model_path, "best") + '/' + '{epoch}-{avg_val_loss:.2f}-{validation_acc:.2f}',
        save_top_k = 10,
        monitor = 'Loss_G',
        mode = 'min',
        # every_n_val_epochs=1,
        every_n_train_steps = 1,
        save_last = True
    )

    model = Generator(args)
    # model = Generator_gan(args)
    # model = Generator_multiinput(args)

    dm = DataModule(data_dir=args.data_dir, test_id=args.test_id, batch_size=args.batch_size, features=args.features, feature_dim=args.feature_dim, opts=args,
                    session_id=args.session_id, purpose=args.purpose, shuffle=True)
    # dm = DataModule_session(data_dir=args.data_dir, test_id=args.test_id, batch_size=args.batch_size, features=args.features, feature_dim=args.feature_dim, opts=args,
    #                 session_id=args.session_id, purpose=args.purpose, shuffle=True)
    print('data prepared')
    # print('preparing data')
    # dm.prepare_data()
    # dm.setup()

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=normalize_trainer_devices(args.accelerator, args.devices),
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        fast_dev_run=args.fast_dev_run,
        profiler=args.profiler,
        benchmark=args.benchmark,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
        logger=False,
    )
    if args.purpose == "train":
        trainer.fit(model, dm)
        torch.save(model.state_dict(),os.path.join(args.model_path, "modelpt.pth"))
        # model.load_state_dict(torch.load(os.path.join(args.model_path, "modelpt.pth")))
        gen_latent(data_loader=dm.train_dataloader(), model=model, save_dir=args.latent_path, role='train')  
        # gen_latent_multi(data_loader=dm.train_dataloader(), model=model, save_dir=args.latent_path, role='train')
        draw_reconsmap(args, 'train', ['final'])
    else:
        model.load_state_dict(torch.load(os.path.join(args.model_path, "modelpt.pth")))
        trainer.test(model, datamodule=dm)
        if args.purpose == "test":
            gen_latent(data_loader=dm.test_dataloader(), model=model, save_dir=args.latent_path, role='test', index='final')  
            # gen_latent_multi(data_loader=dm.test_dataloader(), model=model, save_dir=args.latent_path, role='test', index='final')
            draw_reconsmap(args, 'test')


def get_best_model(model_path, use_best=True):
    if not use_best:
        checkpoint_path = os.path.join(model_path, "final_model.ckpt")
    else:
        best_model_path = os.path.join(model_path, "best")
        best_model_path = glob.glob(os.path.join(best_model_path, "*.ckpt"))
        best_model_path = sorted(best_model_path, key=os.path.getmtime)
        checkpoint_path = best_model_path[-1]
        return checkpoint_path


def create_dirs(dirs_list):
    for dir in dirs_list:
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        # print(f'DIR {dir} CREATED')


def config_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(config, data_type, data_root=None, data_dir=None):
    if data_dir:
        return data_dir
    if data_root:
        return os.path.join(data_root, config.train_dir_name[data_type])
    return config.train_dir[data_type]


def apply_data_type_defaults(args):
    feature_counts = {
        'resample171_data_half': 171,
        'smooth_rms_half': 240,
        'resample_stft_half': 64 * 4,
        'stft_half': 256,
        'stft_half_half': 129,
        'stft_triple_half': 384,
        'StftFeature': 260,
    }
    if args.data_type in feature_counts:
        args.features = list(range(feature_counts[args.data_type]))
    else:
        args.features = [int(i) for i in args.features]
    args.feature_dim = 1
    args.encoder_channels = [len(args.features) * args.feature_dim, 512, 256, 128]
    args.test_id = [int(i) for i in args.test_id]


def normalize_trainer_devices(accelerator, devices):
    if accelerator == "cpu" and isinstance(devices, list):
        return devices[0] if len(devices) == 1 else len(devices)
    return devices


if __name__ == '__main__':
    config = Config()
    parser = ArgumentParser()
    VERSION = 0
    parser.add_argument('--test_id', type=int, nargs='+', default=[1,2,3,4],
                        help='Loaded subject ids. Data files are expected under subjectXX_sessionY folders.')
    parser.add_argument("--session_id", type=int, default=1,
                        help='Training session id. DataModule_session uses the other session as test data.')
    parser.add_argument('--trail_id', type=int, default=1,
                        help='Experiment/trial id used to namespace outputs. Kept as trail_id for compatibility.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--root_dir', type=str, default=os.environ.get('HAND_DT_ROOT', './runs'),
                        help='Root directory for models, outputs, and latent features.')
    parser.add_argument('--data_root', type=str, default=os.environ.get('HAND_DT_DATA_ROOT'),
                        help='Optional root containing preprocessed data folders listed in settings.py.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Explicit preprocessed data directory. Overrides --data_root and --data_type mapping.')
    parser.add_argument("--class_num", type=int, default=11, help='Number of gesture classes.')
    parser.add_argument("--data_type", type=str, default='unslice_features_half',
                        choices=list(config.train_dir.keys()),
                        help='Preprocessed data type key defined in settings.py.')
    parser.add_argument("--features", type=int, nargs='+', default=[0],
                        help='Feature indices used when data_type is unslice_features_half or another custom feature set.')
    parser.add_argument("--feature_dim", type=int, default=1, help='Feature dimension per selected feature.')
    parser.add_argument("--encoder_channels", type=int, nargs='+', default=[1, 512, 256, 128],
                        help='Encoder channel sizes. The input channel is reset from selected features.')
    parser.add_argument("--decoder_type", type=str, default='vae', choices=['vae', 'orig'],
                        help='Decoder architecture: vae for transpose-conv decoder, orig for upsample-conv decoder.')
    parser.add_argument("--clf_methods", type=str, nargs='+', default=['knn', 'rf'],
                        help='Classifier methods used by downstream evaluation scripts.')

    # Model specific arguments etc. layer dim, num layers, learning_rate`
    parser.add_argument("--lr", type=float, default=2e-3, help='Learning rate.')
    parser.add_argument("--scheduler_t_max", type=int, default=40, help='Reserved scheduler T_max parameter.')

    parser.add_argument('--accelerator', type=str, default='gpu', choices=['cpu', 'gpu', 'auto'],
                        help='PyTorch Lightning accelerator.')
    parser.add_argument('--devices', type=int, nargs='+', default=[0],
                        help='Device ids for Lightning. Use --accelerator cpu --devices 1 for CPU.')
    parser.add_argument("--batch_size", type=int, default=512, help='Batch size. 512 was used for inter-subject runs.')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--max_epochs', type=int, default=400, help='Maximum training epochs.')
    parser.add_argument('--min_epochs', type=int, default=1, help='Minimum training epochs.')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5, help='Validation interval in epochs.')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a one-batch Lightning smoke test.')
    parser.add_argument('--profiler', default=False, help='Lightning profiler setting.')
    parser.add_argument('--benchmark', action='store_true', help='Enable cuDNN benchmark mode.')
    parser.add_argument('--dropout_prob', type=float, default=.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lamda', type=float, default=0.5, help='loss_trip=lamda*loss_p + (1-lamda)*loss_s')
    parser.add_argument('--lamda1', type=float, default=0.5, help='lamda1*(loss_p+loss_s) + lamda2*Loss_cross_recon')
    parser.add_argument('--lamda2', type=float, default=1, help='lamda1*(loss_p+loss_s) + lamda2*Loss_cross_recon')
    parser.add_argument('--log_every_n_steps', type=int, default=1)


    parser.add_argument('--purpose', type=str, default="test",
                        choices=['train', 'test', 'test_trainset', 'test_allset'],
                        help='train fits the model; test loads modelpt.pth and exports test latents.')
    parser.add_argument('--mode', type=str, default="run", choices=['run', 'getpath'],
                        help='Use getpath to print the terminal log directory and exit.')

    parser = Generator.add_model_specific_args(parser)
    args = parser.parse_args()

    rootdir = os.path.abspath(args.root_dir)
    args.data_dir = resolve_data_dir(config, args.data_type, args.data_root, args.data_dir)
    args.default_root_dir = f'{rootdir}/models/trail_{args.trail_id}/test_{args.test_id[0]}'
    args.model_path = f'{rootdir}/models/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'
    args.output_path = f'{rootdir}/outputs/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'
    args.output_log = f'{rootdir}/outputs/terminal/trail_{args.trail_id}/'
    args.acc_path = f'{rootdir}/outputs/trail_{args.trail_id}'
    args.latent_path = f'{rootdir}/latent_features/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'

    create_dirs([args.model_path, args.output_path, args.latent_path, args.output_log])
    config_random_seed(args.seed)
    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    apply_data_type_defaults(args)

    print(f'------------------{args.features[-1]}--------------------')
    main(args)

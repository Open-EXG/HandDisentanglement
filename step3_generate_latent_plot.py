import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pathlib
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import random

from step0_main_code import apply_data_type_defaults, resolve_data_dir
from models import Generator
from data_module import  DataModule
from settings import Config


def _gen_latent(data_loader,model,save_dir,role,index='final'):
    xp_total=[]
    xs_total=[]
    yp_total=[]
    ys_total=[]
    recon_p_total=[]
    recon_s_total=[]
    for idx, batch in enumerate(data_loader):
        data1 = batch['x'].type(torch.FloatTensor)
        yp = batch["p"].type(torch.LongTensor)
        ys = batch["s"].type(torch.LongTensor)

        xp=model.encoder_p(data1)
        xs=model.encoder_s(data1)
        recon_p = model.decoder(torch.cat([xp, torch.zeros_like(xs)], dim=1))
        recon_s = model.decoder(torch.cat([torch.zeros_like(xp), xs], dim=1))

        xp=xp.detach().cpu().numpy()
        xs=xs.detach().cpu().numpy()
        recon_p=recon_p.detach().cpu().numpy()
        recon_s=recon_s.detach().cpu().numpy()

        xp_total.append(xp)
        xs_total.append(xs)
        yp_total.append(yp)
        ys_total.append(ys)
        recon_p_total.append(recon_p)
        recon_s_total.append(recon_s)
    xp_total=np.vstack(xp_total) #(240,128,2,2) 解耦后
    xs_total=np.vstack(xs_total) #(240,128,2,2)
    yp_total=np.hstack(yp_total) #动作标签
    ys_total=np.hstack(ys_total) #sub_id 
    recon_p_total=np.vstack(recon_p_total) #p分支
    recon_s_total=np.vstack(recon_s_total) #s分支
    np.savez(os.path.join(save_dir,f"{role}_{index}.npz"),xp=xp_total,xs=xs_total,yp=yp_total,ys=ys_total,recon_p=recon_p_total,recon_s=recon_s_total)

def create_dirs(dirs_list):
    for dir in dirs_list:
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)


def config_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)    
    random.seed(seed)


def main(args):
    print(f'--generating recons features--')
    # model= Generator_gan(args)
    model= Generator(args)
    model.load_state_dict(torch.load(os.path.join(args.model_path, f"modelpt.pth")))
    model.eval()
    save_dir = args.latent_path
 
    pathlib.Path(save_dir).mkdir(exist_ok=True,parents=True)
    dm = DataModule(data_dir=args.data_dir, test_id=args.test_id, batch_size=args.batch_size, features=args.features, feature_dim=args.feature_dim, opts=args,
                    session_id=args.session_id, purpose=args.purpose, shuffle=False)

    dm.prepare_data()
    dm.setup()
    test_dataloader=dm.test_dataloader()
    train_dataloader=dm.train_dataloader()

    _gen_latent(test_dataloader,model,save_dir,role="test")
    _gen_latent(train_dataloader,model,save_dir,role="train")
    print(f"{args.test_id} done")

def main_plot(args, purpose, index_list=['final']):
    print(f'--plotting recons features--')

    for index in index_list:
        with np.load(os.path.join(args.latent_path,f"{purpose}_{index}.npz")) as f:
            recon_s = f['recon_s']  #(240,4,16,16)
            recon_s=recon_s[:,0,:,:]

            recon_p = f['recon_p']  #(240,4,16,16)
            recon_p=recon_p[:,0,:,:]
            
            p_data_index=f["yp"]
            s_data_index=f["ys"]

        unique_s_values = np.unique(s_data_index)
        unique_p_values = np.unique(p_data_index)

        # 初始化累加器
        accumulators = {
            (p_label, s_value): {'recon_s': np.zeros((16, 16)),
                                 'recon_p': np.zeros((16, 16)),
                                 'count': 0} 
            for p_label in unique_p_values for s_value in unique_s_values
        }

        for i in range(recon_s.shape[0]):
            recon_s_temp = recon_s[i]
            recon_p_temp = recon_p[i]
            p=p_data_index[i]
            s=s_data_index[i]

            # # 测试单个样本
            # recon_p_zoom = zoom(recon_p_temp,(10,10),order=3)
            # plt.imshow(recon_p_zoom, cmap='jet')
            # print(os.getcwd())
            # plt.savefig(f'./hand_DT/test.png', dpi=300)
            # plt.show()
            # return

            # 累加相同 p_data_index 和 s_data_index 的图像
            acc_key = (p, s)
            accumulators[acc_key]['recon_s'] += recon_s_temp
            accumulators[acc_key]['recon_p'] += recon_p_temp
            accumulators[acc_key]['count'] += 1    

        # 创建图形和子图，每个 p_data_index 对应一行，每个 s_data_index 对应一列
        # fig, axes = plt.subplots(len(unique_p_values), len(unique_s_values), figsize=(len(unique_s_values) * 5, len(unique_p_values) * 5))
        fig, axes = plt.subplots(len(unique_p_values), 1, figsize=(5, len(unique_p_values) * 5))
        # 确保 axes 是二维的，方便索引
        if len(unique_p_values) == 1:
            axes = np.expand_dims(axes, 0)
        if len(unique_s_values) == 1:
            axes = np.expand_dims(axes, 1)

        # 对每个 p_data_index 和 s_data_index 的组合求平均并绘制热力图
        for i, p_label in enumerate(unique_p_values):
            recon_p_avg_total = np.zeros_like(accumulators[acc_key]['recon_p'])
            for j, s_value in enumerate(unique_s_values):
                acc_key = (p_label, s_value)
                c = 0 # 跳过的被试数
                if accumulators[acc_key]['count'] == 0:
                    c+=1
                    continue
                recon_s_avg = accumulators[acc_key]['recon_s'] / accumulators[acc_key]['count']
                recon_p_avg = accumulators[acc_key]['recon_p'] / accumulators[acc_key]['count']
                recon_p_avg_total += recon_p_avg

                recon_s_avg = zoom(recon_s_avg,(10,10),order=3)
                recon_p_avg = zoom(recon_p_avg,(10,10),order=3)
                
                # # 绘制 s 分支和 p 分支
                # ax_recon_s = axes[i, j]
                # cax_recon_s = ax_recon_s.imshow(recon_s_avg, cmap='jet')
                # ax_recon_s.set_title(f"Recon_s Label {p_label} Sub_id {s_value}")
                # if j == 0:  
                #     fig.colorbar(cax_recon_s, ax=ax_recon_s)
                # ax_recon_p = axes[i, j + len(unique_s_values)]  
                # cax_recon_p = ax_recon_p.imshow(recon_p_avg, cmap='jet')
                # ax_recon_p.set_title(f"Recon_p Label {p_label} Sub_id {s_value}")
                # if j == 0:  
                #     fig.colorbar(cax_recon_p, ax=ax_recon_p)

                # # 仅绘制 p 分支，不加图例
                # ax_recon_p = axes[i, j]
                # cax_recon_p = ax_recon_p.imshow(recon_p_avg, cmap='jet')
                # ax_recon_p.set_title(f"Recon_p Label {p_label} Sub_id {s_value}")

            # 仅绘制 p 分支， 对所有被试求平均
            ax_recon_p = axes[i]
            recon_p_avg_total = recon_p_avg_total/(j+1-c)
            recon_p_avg_total = zoom(recon_p_avg_total,(10,10),order=3)
            cax_recon_p = ax_recon_p.imshow(recon_p_avg_total, cmap='jet')
            ax_recon_p.set_title(f"Recon_p Label {p_label} Sub_avg")
                
        # 显示图形
        plt.tight_layout()
        # plt.savefig(f'{args.output_path}/recons_figure_{purpose}_{index}_avg.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{args.output_path}/recons_figure_{purpose}_{index}_avg.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    config = Config()
    parser = ArgumentParser()
    VERSION = 0
    parser.add_argument('--test_id', type=int, nargs='+', default=[1,2,3,4],
                        help='Subject ids used in path naming.')
    parser.add_argument("--session_id", type=int, default=1, help='Session id.')
    parser.add_argument('--root_dir', type=str, default=os.environ.get('HAND_DT_ROOT', './runs'),
                        help='Root directory created by step0_main_code.py.')
    parser.add_argument('--data_root', type=str, default=os.environ.get('HAND_DT_DATA_ROOT'),
                        help='Optional root containing preprocessed data folders listed in settings.py.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Explicit data directory. Overrides --data_root and --data_type mapping.')
    parser.add_argument('--output_path', type=str, default="", help='Optional explicit plot output directory.')
    parser.add_argument('--trail_id', type=int, default=12, help='Experiment/trial id.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("--class_num", type=int, default=10, help='Number of gesture classes.')
    parser.add_argument("--data_type", type=str, default='stft_half', choices=list(config.train_dir.keys()))
    parser.add_argument("--decoder_type", type=str, default='vae')  # 'vae': Dacoder_vae  'orig': Decoder
    parser.add_argument("--features", type=int, nargs='+', default=[0,1,24,25])
    parser.add_argument("--feature_dim", type=int, default=1)

    # Model specific arguments etc. layer dim, num layers, learning_rate`
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--scheduler_t_max", type=int, default=40)

    # trainer level args
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--min_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--fast_dev_run', action='store_true')  # run one epoch to output
    parser.add_argument('--profiler', default=False)  # can change to advanced profiler
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--purpose', type=str, default="test",
                        choices=['train', 'test', 'test_trainset', 'test_allset'])
    parser.add_argument('--mode', type=str, default="plt", choices=['gen', 'plt', 'both'],
                        help='gen exports latent npz, plt plots existing latent files, both does both.')

    parser = Generator.add_model_specific_args(parser)
    # parser = Generator_gan.add_model_specific_args(parser)
    args = parser.parse_args()

    # create path
    rootdir = os.path.abspath(args.root_dir)
    args.data_dir = resolve_data_dir(config, args.data_type, args.data_root, args.data_dir)
    args.model_path = f'{rootdir}/models/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'
    args.output_path = args.output_path or f'{rootdir}/outputs/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'
    args.output_log = f'{rootdir}/outputs/terminal/trail_{args.trail_id}/'
    args.acc_path = f'{rootdir}/outputs/trail_{args.trail_id}'
    args.latent_path = f'{rootdir}/latent_features/trail_{args.trail_id}/test_{args.test_id[0]}/session_{args.session_id}/'

    print(f'------------------{args.features}--------------------')
    create_dirs([args.latent_path, args.output_path, args.output_log])
    config_random_seed(args.seed)
    apply_data_type_defaults(args)

    if args.mode == 'gen':
        main(args)
    elif args.mode == 'plt':
        main_plot(args, 'train', index_list=['final'])
    else:
        main(args)
        main_plot(args, 'train')

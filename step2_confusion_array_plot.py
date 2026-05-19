from scipy.signal import correlate2d
from argparse import ArgumentParser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

XY_AXIS = ['1','2','3','4','5','6','7','8','9', '10', 'avg']

def cal_correlate2d(latent_path, test_id, session_id, trial_id, gestures):
    corr2d = np.eye(len(gestures)+1)
    train_x = []
    train_y = []
    for i in test_id:
        for j in session_id:
            tmp_x = []
            for t in trial_id:
                with np.load(f"{latent_path}/trail_{t}/test_{i}/session_{j}/train_final.npz") as f:
                    xp = f["recon_p"]
                    yp = f["yp"]                    
                    tmp_x.append(xp)
            train_x.append(np.hstack(tmp_x))
            train_y.append(yp)
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)


    for g in gestures:
        ges_idx1 = [i for (i,ele) in enumerate(train_y) if ele==g]
        data1 = train_x[ges_idx1]
        data1_mean = np.mean(data1,axis=0)
        gestures_left = [i for i in gestures if i!=g]
        for o in gestures_left:
            ges_idx2 = [i for (i,ele) in enumerate(train_y) if ele==o]
            data2 = train_x[ges_idx2]
            data2_mean = np.mean(data2,axis=0)
            corr = np.corrcoef(data1_mean.flatten(), data2_mean.flatten())
            corr2d[g,o] = abs(corr[0,1])
        # ges_idx3 = [i for (i,ele) in enumerate(train_y) if ele!=g]
        # data3 = train_x[ges_idx3]
        # data3_mean = np.mean(data3,axis=0)
        # corr = np.corrcoef(data1_mean.flatten(), data3_mean.flatten())
        # corr2d[g,-1] = %abs(corr[0,1])
        # corr2d[-1,g] = %abs(corr[0,1])
        corr2d[g,-1] = np.mean(corr2d[g,gestures_left])
        corr2d[-1,g] = np.mean(corr2d[g,gestures_left])
    corr2d[-1,-1] = np.mean(corr2d[:-2,-1])
    return corr2d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./runs',
                        help='Root directory created by step0_main_code.py.')
    parser.add_argument('--trial_id', type=int, nargs='+', default=[5],
                        help='Trial ids to concatenate.')
    parser.add_argument('--test_id', type=int, nargs='+', default=[1],
                        help='Subject ids to include.')
    parser.add_argument('--session_id', type=int, nargs='+', default=[1],
                        help='Session ids to include.')
    parser.add_argument('--gestures', type=int, nargs='+', default=list(range(10)),
                        help='Gesture labels to plot.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path.')
    args = parser.parse_args()

    latent_path = f'{args.root_dir}/latent_features'
    corr2d = cal_correlate2d(latent_path=latent_path, test_id=args.test_id,
                             session_id=args.session_id, trial_id=args.trial_id,
                             gestures=args.gestures)
    mask_map = np.eye(corr2d.shape[0]).astype(bool)
    mask_map[-1,-1] = False

    sns.set_context({"figure.figsize":(8,8)})
    labels = [str(i + 1) for i in args.gestures] + ['avg']
    ax = sns.heatmap(data=corr2d, xticklabels=labels, yticklabels=labels,
                     square=True, fmt='0.2f', mask=mask_map, vmax=1, vmin=0,
                     annot=True, cmap="GnBu", cbar_kws={'shrink': 0.81})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    figure = ax.get_figure()
    output = args.output or f'test_sns_heatmap_{args.trial_id}.png'
    figure.savefig(output, dpi=300)

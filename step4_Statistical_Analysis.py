from argparse import ArgumentParser
import os
import seaborn as sns
from scipy.stats import ttest_ind
from scipy import stats
import numpy as np

def stattest(data1,data2):
    normindex=True
    testnormp1=stats.shapiro(data1).pvalue
    testnormp2=stats.shapiro(data2).pvalue
    # 如果不满足就直接不进行ttest
    if testnormp1<0.05 or testnormp2<0.05 :normindex=False
    # 看是不是正态性来决定用什么检验方式
    test1=np.array(data1)[np.where((np.isnan(data1)+np.isnan(data1))==False)]
    test2=np.array(data2)[np.where((np.isnan(data2)+np.isnan(data2))==False)]

    # 开始检验
    if normindex:
        if np.mean(test1)<=np.mean(test2):
            p_AB = -ttest_ind(test1,test2,alternative='less').pvalue
        else:
            # 因为要论证的是CNN小于其他method，所以
            p_AB = ttest_ind(test1, test2,alternative='greater').pvalue
    else:
        if np.mean(test1)<=np.mean(test2):
            p_AB= -stats.wilcoxon(test1,test2,correction=False,alternative='less').pvalue
        else:
            p_AB= stats.wilcoxon(test1,test2,correction=False,alternative='greater').pvalue

    # p-value如果为-，说明第一个数据小于第二个
    return p_AB


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, nargs='+', default=['./runs'],
                        help='One or more run roots containing outputs/*_acc.npy.')
    parser.add_argument('--trial_id', type=int, nargs='+', required=True,
                        help='Trial ids to compare. Use one id per condition.')
    parser.add_argument('--method', type=str, default='rf',
                        help='Classifier method name used in *_acc.npy files.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output heatmap image path.')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Axis labels. Defaults to trial ids.')
    args = parser.parse_args()

    METHOD = args.method
    data = []
    roots = args.root_dir if len(args.root_dir) > 1 else args.root_dir * len(args.trial_id)
    for root, trial_idx in zip(roots, args.trial_id):
        acc_files = [
            f'{root}/outputs/{[trial_idx]}_{METHOD}_1_acc.npy',
            f'{root}/outputs/{[trial_idx]}_{METHOD}_2_acc.npy',
            f'{root}/outputs/{trial_idx}_{METHOD}_1_acc.npy',
            f'{root}/outputs/{trial_idx}_{METHOD}_2_acc.npy',
        ]
        loaded = [np.load(path) for path in acc_files if os.path.exists(path)]
        if not loaded:
            raise FileNotFoundError(f'No accuracy files found for trial {trial_idx} under {root}/outputs')
        data.append(np.mean(np.array(loaded), axis=0))
    
    statistic_results = {}
    statistic_array = np.zeros((len(data),len(data)))
    for idx1, data1 in enumerate(data):
        for idx2, data2 in enumerate(data):
            if idx1 == idx2:
                continue
            p_value = stattest(data1=data1, data2=data2)
            statistic_results[f'{idx1}_{idx2}'] = p_value
            statistic_array[idx1,idx2] = p_value
            statistic_array[idx2,idx1] = p_value

    XY_AXIS = args.labels or [str(i) for i in args.trial_id]
    mask_map = np.eye(len(data)).astype(bool)
    sns.set_context({"figure.figsize":(8,8)})
    ax = sns.heatmap(data=statistic_array, xticklabels=XY_AXIS, yticklabels=XY_AXIS, square=True, fmt='0.2f',
                    mask=mask_map, vmax=1, vmin=-1, annot=True, cmap="RdBu", cbar_kws={'shrink': 0.81})  # GnBu
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    figure = ax.get_figure()
    output = args.output or f'statistic_{METHOD}.png'
    figure.savefig(output, dpi=300)

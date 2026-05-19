from argparse import ArgumentParser
from copy import deepcopy
import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

import pytorch_lightning as pl

from scipy.ndimage import zoom
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import yaml


def _fit_tsne(data, n_components=2, perplexity=50, max_iter=3000):
    if data.shape[0] <= 1:
        return np.zeros((data.shape[0], n_components))

    tsne_kwargs = {
        "n_components": n_components,
        "init": "pca",
        "random_state": 0,
        "perplexity": min(perplexity, data.shape[0] - 1),
        "early_exaggeration": 100,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = max_iter
    else:
        tsne_kwargs["n_iter"] = max_iter
    return TSNE(**tsne_kwargs).fit_transform(data)


def gen_latent(data_loader,model,save_dir,role='train',index='final'):
    device = next(model.parameters()).device
    model.eval()
    xp_total=[]
    xs_total=[]
    yp_total=[]
    ys_total=[]
    recon_p_total=[]
    recon_s_total=[]
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            data1 = batch['x'].float().to(device)
            yp = batch["p"].long()
            ys = batch["s"].long()

            xp=model.encoder_p(data1)
            xs=model.encoder_s(data1)
            recon_p = model.decoder(torch.cat([xp, torch.zeros_like(xs)], dim=1))
            recon_s = model.decoder(torch.cat([torch.zeros_like(xp), xs ], dim=1))

            xp=xp.detach().cpu().numpy()
            xs=xs.detach().cpu().numpy()
            recon_p=recon_p.detach().cpu().numpy()
            recon_s=recon_s.detach().cpu().numpy()

            xp_total.append(xp)
            xs_total.append(xs)
            yp_total.append(yp.cpu().numpy())
            ys_total.append(ys.cpu().numpy())
            recon_p_total.append(recon_p)
            recon_s_total.append(recon_s)
    xp_total=np.vstack(xp_total) #(240,128,2,2) 解耦后
    xs_total=np.vstack(xs_total) #(240,128,2,2)
    yp_total=np.hstack(yp_total) #动作标签
    ys_total=np.hstack(ys_total) #sub_id 
    recon_p_total=np.vstack(recon_p_total) #p分支
    recon_s_total=np.vstack(recon_s_total) #s分支
    np.savez(os.path.join(save_dir,f"{role}_{index}.npz"),xp=xp_total,xs=xs_total,yp=yp_total,ys=ys_total,recon_p=recon_p_total,recon_s=recon_s_total)

def gen_latent_multi(data_loader,model,save_dir,role='train',index='final'):
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

        xp=torch.cat([model.encoder_p_stft(data1[:,:256,:,:]),model.encoder_p_fea(data1[:,256:,:,:])], dim=1)
        xs=torch.cat([model.encoder_s_stft(data1[:,:256,:,:]),model.encoder_s_fea(data1[:,256:,:,:])], dim=1)
        recon_p = model.decoder(torch.cat([xp, torch.zeros_like(xs)], dim=1))
        recon_s = model.decoder(torch.cat([torch.zeros_like(xp), xs ], dim=1))

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

def draw_reconsmap(args, purpose, index_list=['final']):
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

            # 累加相同 p_data_index 和 s_data_index 的图像
            acc_key = (p, s)
            accumulators[acc_key]['recon_s'] += recon_s_temp
            accumulators[acc_key]['recon_p'] += recon_p_temp
            accumulators[acc_key]['count'] += 1    

        # 创建图形和子图，每个 p_data_index 对应一行，每个 s_data_index 对应一列
        fig, axes = plt.subplots(len(unique_p_values), len(unique_s_values), figsize=(len(unique_s_values) * 5, len(unique_p_values) * 5))
        # 确保 axes 是二维的，方便索引
        if len(unique_p_values) == 1:
            axes = np.expand_dims(axes, 0)
        if len(unique_s_values) == 1:
            axes = np.expand_dims(axes, 1)

        # 对每个 p_data_index 和 s_data_index 的组合求平均并绘制热力图
        for i, p_label in enumerate(unique_p_values):
            # for j, s_value in enumerate(unique_s_values):
            for j, s_value in enumerate(unique_s_values):
                acc_key = (p_label, s_value)
                recon_s_avg = accumulators[acc_key]['recon_s'] / accumulators[acc_key]['count']
                recon_p_avg = accumulators[acc_key]['recon_p'] / accumulators[acc_key]['count']
                recon_s_avg = zoom(recon_s_avg,(10,10),order=3)
                recon_p_avg = zoom(recon_p_avg,(10,10),order=3)

                ax_recon_p = axes[i, j]  
                cax_recon_p = ax_recon_p.imshow(recon_p_avg, cmap='jet')
                ax_recon_p.set_title(f"Recon_p Label {p_label} Sub_id {s_value}")
                # if j == 0:  
                #     fig.colorbar(cax_recon_p, ax=ax_recon_p)
                
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
                
        # 显示图形
        plt.tight_layout()
        plt.savefig(f'{args.output_path}/recons_figure_{purpose}_{index}.png', dpi=300, bbox_inches='tight')

def calculate_acc(test_id, session_id, data_dir, save_dir, methods):
    dim=512
    result_tmp = {}
    with np.load(f'{data_dir}/train.npz') as f:
        xp = f["xp"]
        xp = xp.reshape((-1,dim))  # xp=xp.reshape((-1,256))
        xs = f["xs"]
        yp = f["yp"]
        print(np.unique(yp))
        ys = f["ys"]
    train_x=xp
    train_y=yp
    with np.load(f'{data_dir}/test.npz') as f:
        xp=f["xp"]
        xp=xp.reshape((-1,dim))  # xp=xp.reshape((-1,256))
        xs=f["xs"]
        yp = f["yp"]
        ys = f["ys"]
    test_x=xp
    test_y=yp
    for method in methods:
        print(f"{method} fitting")
        clf=classify(train_x,train_y,test_x,test_y,type=method)
        output=clf.predict(test_x)
        print("fitting end")
        acc = accuracy_score(test_y, output)
        result_tmp[f'{method}'] = acc
        print(acc)
    
    # 结果存储为字典
    # {'{test1}_{session1}': {'knn': acc, 'svm': acc, 'rf': acc},
    #  '{test1}_{session2}': {'knn': acc, 'svm': acc, 'rf': acc},
    #  ... ... }
    yamlfile = os.path.join(save_dir, 'acc_results.yaml')
    if not os.path.exists(yamlfile):  # 若文件不存在，新建空字典
        with open(yamlfile, 'w', encoding='etf-8') as f:
            yaml.safe_dump({}, f)
    with open(yamlfile, 'r', encoding='utf-8') as f:  # 读取字典
        acc_result = yaml.safe_load(f)
    
    acc_result[f'{test_id}_{session_id}'] = result_tmp  # 添加新元素

    with open(yamlfile, 'w') as f:
        yaml.safe_dump(acc_result, f)

def weights_init_kaiming(m):  # 权重初始化：权重是通过线性层（卷积或全连接）torch.nn.xxx 隐性确定的，mode=fan_in; 通过创建随机矩阵显式创建权重 torch.randn()，mode=fan_out
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        # init.normal_(m.weight.data, 1.0, 0.02)
        # init.constant_(m.bias.data, 0.0)
        print("instance")

def Dict2Obj(d: dict):
    if not isinstance(d, dict):
            return d
    x = ArgumentParser()
    for k, v in d.items():
        x.add_argument(f'--{k}', type=type(v), default=v)
    return x.parse_args()

def classify(train_x, train_y, test_x, test_y, type="rf"):
   if type=="rf":
       clf = RandomForestClassifier(n_estimators=1000, random_state=0)
       clf.fit(train_x, train_y)

   elif type=="svm":
       clf = svm.LinearSVC()
       clf.fit(train_x, train_y)

   elif type=="knn":
       clf = KNeighborsClassifier(n_neighbors=1)
       clf.fit(train_x, train_y)

   return clf


class Encoder_p(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False):
        super(Encoder_p, self).__init__()
        model = []
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[1]),
                nn.LeakyReLU(0.2),
            ))
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[2]),
                nn.LeakyReLU(0.2),
            ))
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[2],
                    out_channels=channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[3]),
                nn.LeakyReLU(0.2),
            ))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Encoder_s(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False):
        super(Encoder_s, self).__init__()
        model = []
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[1]),
                nn.LeakyReLU(0.2),
            ))
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[2]),
                nn.LeakyReLU(0.2),
            ))
        model.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels[2],
                    out_channels=channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Dropout(.1),
                nn.InstanceNorm2d(channels[3]),
                nn.LeakyReLU(0.2),
            ))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Decoder, self).__init__()
        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)
        for i in range(len(channels) - 1):

            model.append(nn.Upsample(scale_factor=2, mode='nearest'))

            model.append(nn.ReflectionPad2d(pad))

            model.append(nn.Conv2d(channels[i], channels[i + 1],
                                   kernel_size=kernel_size, stride=1))
            model.append(nn.Dropout(p=0.2))

            if not i == len(channels) - 2:
                model.append(acti)  # whether to add tanh in tha last layer
                # model.append(nn.Dropout(p=0.2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class Decoder_vae(nn.Module):
    def __init__(self, channels=[256,128,64,128]) -> None:
        super(Decoder_vae, self).__init__()

        modules = []
        # for i in range(len(channels) - 2):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
        #             nn.BatchNorm2d(channels[i + 1]),
        #             nn.LeakyReLU())
        #     )

        # self.model = nn.Sequential(*modules)

        self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channels[2]),
                nn.LeakyReLU())

        self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(64, channels[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.Conv2d(64, 32,  kernel_size=3, padding=1),
                nn.Tanh())
        
        self.embed_data = nn.Conv2d(channels[-1], channels[-1], kernel_size=1)

    def forward(self, x):
        # x = self.model(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_layer(x)
        return self.embed_data(x)


class Discriminator_gan(nn.Module):
    def __init__(self, image_size, channels, num_classes=0) -> None:
        super(Discriminator_gan, self).__init__()

        # self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    # def forward(self, inputs: torch.Tensor, labels: list = None) -> torch.Tensor:
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (N*C*H*W).
        """
        inputs = torch.flatten(inputs, 1)
        # conditional = self.label_embedding(labels)
        # conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        # out = self.main(conditional_inputs)
        out = self.main(inputs)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    

class Generator(pl.LightningModule):
    def __init__(self, args):
        super(Generator, self).__init__()

        if isinstance(args, dict):
            args = Dict2Obj(args)

        self.args = args
        self.encoder_p = Encoder_p(channels=args.encoder_channels, kernel_size=4)
        self.encoder_s = Encoder_s(channels=args.encoder_channels, kernel_size=4)
        if args.decoder_type == 'vae':
            self.decoder = Decoder_vae(channels=[256, 128, 64, len(args.features)*args.feature_dim])
        elif args.decoder_type == 'orig':
            self.decoder = Decoder(channels=[256, 128, 64, len(args.features)*args.feature_dim])
        # for m in self.modules():
        #     weights_init_kaiming(m)
        self.test_outputs = []
        self.save_hyperparameters(args)

    def forward(self, x):
        p = self.encoder_p(x)
        # print("p shape",p.shape)
        s = self.encoder_s(x)
        # print("s shape", s.shape)
        # print("concat shape",torch.cat([p, s], dim=1).shape)

        x = self.decoder(torch.cat([p, s], dim=1))
        return x, p, s

    def cross(self, x_p1_sk, x_pk_s1):  # 交叉重构
        p1 = self.encoder_p(x_p1_sk)
        sk = self.encoder_s(x_p1_sk)
        pk = self.encoder_p(x_pk_s1)
        s1 = self.encoder_s(x_pk_s1)
        out_p1_s1 = self.decoder(torch.cat([p1, s1], dim=1))
        return out_p1_s1
    
    @property
    def loss(self):
        return {
            "reconstruction_criterion": nn.MSELoss(),
            "trip_criterion": nn.TripletMarginLoss(margin=0.3),
            "p_criterion": nn.CrossEntropyLoss(),
            "s_criterion": nn.CrossEntropyLoss()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.scheduler_t_max, eta_min=0, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=200)
        return [optimizer], [scheduler]

        # return optimizer

    def training_step(self, batch, batch_idx):
        data1 = batch['x'].float().to(self.device)
        p1_id_gt = batch["p"].long().to(self.device)
        s1_id_gt = batch["s"].long().to(self.device)
        data1_recon, p1, s1 = self(data1)
        p1 = p1.reshape(p1.shape[0], -1)
        s1 = s1.reshape(s1.shape[0], -1)

        ## reconstruction loss
        Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)

        # # triplet loss pattern
        # # Loss_trip_p = 0
        Loss_trip_p = 0
        Loss_trip_p_number = 0
        achor_p=[]
        positive_p=[]
        negative_p=[]
        for i in np.unique(p1_id_gt.data.cpu()):
            for j in np.unique(s1_id_gt.data.cpu()):
                data_index_pisj = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)
                data_index_pisk = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                data_index_pksl = ((p1_id_gt != i)).nonzero().squeeze(dim=-1)
                # why dim need to greater than 0, what about [index]?
                if data_index_pisj.dim() > 0:
                    data_len = min(len(data_index_pisj), len(data_index_pisk), len(data_index_pksl))
                    # print(f"find availibale pattern labels --> len:{data_len}")

                    if data_len == 0:
                        continue
                    data_index_pisk = data_index_pisk[0:data_len].squeeze(dim=-1)
                    data_index_pksl = data_index_pksl[0:data_len].squeeze(dim=-1)
                    data_index_pisj = data_index_pisj[0:data_len].squeeze(dim=-1)

                    # print('pisk:{:.0f} pksl:{:.0f} pisj:{:.0f}'
                    #      .format(len(p1[data_index_pisk]),len(p1[data_index_pksl]),len(data_index_pisj)))
                    if data_len < 2:
                        a = p1[data_index_pisj].unsqueeze(dim=0)
                        p = p1[data_index_pisk].unsqueeze(dim=0)
                        n = p1[data_index_pksl].unsqueeze(dim=0)
                    else:
                        a = p1[data_index_pisj]
                        p = p1[data_index_pisk]
                        n = p1[data_index_pksl]
                    achor_p.append(a)
                    positive_p.append(p)
                    negative_p.append(n)
                    # Loss_trip_p_number += 1
                else:
                    continue
        if len(achor_p) == 0 or len(positive_p)==0 or len(negative_p)==0:
            return None
        achor_p=torch.cat(achor_p,dim=0)
        positive_p=torch.cat(positive_p,dim=0)
        negative_p=torch.cat(negative_p,dim=0)
        Loss_trip_p = (self.loss["trip_criterion"](achor_p, positive_p, negative_p))

        # # Triplet loss subjects
        # # Loss_trip_s = 0
        achor_s=[]
        positive_s=[]
        negative_s=[]
        Loss_trip_s = 0
        Loss_trip_s_number = 0
        for i in np.unique(s1_id_gt.data.cpu()):
            for j in np.unique(p1_id_gt.data.cpu()):
                data_index_sipj = ((s1_id_gt == i) * (p1_id_gt == j)).nonzero().squeeze(dim=-1)
                data_index_sipk = ((s1_id_gt == i) * (p1_id_gt != j)).nonzero().squeeze(dim=-1)
                data_index_skpl = ((s1_id_gt != i)).nonzero().squeeze(dim=-1)

                if data_index_sipj.dim() > 0:
                    data_len = min(len(data_index_sipj), len(data_index_sipk), len(data_index_skpl))
                    # print(f"find availibale subject labels --> len:{data_len}")
                    if data_len == 0:
                        continue
                    data_index_sipk = data_index_sipk[0:data_len].squeeze(dim=-1)
                    data_index_skpl = data_index_skpl[0:data_len].squeeze(dim=-1)
                    data_index_sipj = data_index_sipj[0:data_len].squeeze(dim=-1)
                    # print('sipk:{:.0f} skpl:{:.0f} sipj:{:.0f}'
                    #      .format(len(s1[data_index_sipk]),len(s1[data_index_skpl]),len(data_index_sipj)))
                    # print(data_len)
                    if data_len < 2:
                        a = s1[data_index_sipj].unsqueeze(dim=0)
                        p = s1[data_index_sipk].unsqueeze(dim=0)
                        n = s1[data_index_skpl].unsqueeze(dim=0)
                    else:
                        a = s1[data_index_sipj]
                        p = s1[data_index_sipk]
                        n = s1[data_index_skpl]
                    achor_s.append(a)
                    positive_s.append(p)
                    negative_s.append(n)
                    # Loss_trip_s += (self.loss["trip_criterion"](a, p, n))
                    # print("loss trip s:",Loss_trip_s)
                    Loss_trip_s_number += 1
                    # validate_loss += data_len.
                else:
                    continue
        if len(achor_s) == 0 or len(positive_s)==0 or len(negative_s)==0:
            return None
        achor_s=torch.cat(achor_s,dim=0)
        positive_s=torch.cat(positive_s,dim=0)
        negative_s=torch.cat(negative_s,dim=0)
        Loss_trip_s = (self.loss["trip_criterion"](achor_s, positive_s, negative_s))

        # cross_reconstruction
        Loss_cross_recon = 0
        Loss_cross_recon_number = 0
        pi=[]
        sj=[]
        pisj=[]
        for i in np.unique(p1_id_gt.data.cpu()):
            for j in np.unique(s1_id_gt.data.cpu()):
                # data_index = (p1_id_gt==i) and (s1_id_gt==j)
                data_index_p = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                data_index_s = ((s1_id_gt == j) * (p1_id_gt != i)).nonzero().squeeze(dim=-1)
                data_index = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)

                if data_index.dim() > 0:

                    # data_index = random.shuffle(data_index)
                    data_index_p = data_index_p[torch.randperm(len(data_index_p))]
                    data_index_s = data_index_s[torch.randperm(len(data_index_s))]

                    data_len = min(len(data_index), len(data_index_p), len(data_index_s))
                    if data_len < 2:
                        continue
                    data_index_p = data_index_p[0:data_len]
                    data_index_s = data_index_s[0:data_len]
                    data_index = data_index[0:data_len]

                    pi.append(data1[data_index_p])
                    sj.append(data1[data_index_s])
                    pisj.append(data1[data_index])
                    # print('p:{:.0f} s:{:.0f} ps:{:.0f}'
                    #      .format(len(data1[data_index_p]),len(data1[data_index_s]),len(data_index)))
                else:
                    continue
        if len(pi) == 0 or len(sj)==0 or len(pisj)==0:
            return None
        pi=torch.cat(pi,dim=0)
        sj=torch.cat(sj,dim=0)
        pisj=torch.cat(pisj,dim=0)
        outpisj = self.cross(pi, sj)
        Loss_cross_recon=self.loss["reconstruction_criterion"](outpisj, pisj)

        # print("\n")
        # print(f"cross_recon_number:{Loss_cross_recon_number} \n Loss_trip_s_number:{Loss_trip_s_number}  \n Loss_trip_p_number: {Loss_trip_p_number}")
        # Loss_G = Loss_recon + 0.5 * Loss_trip_p * (1. / Loss_trip_p_number) + 0.5 * Loss_trip_s * (
        #             1. / Loss_trip_s_number)
        # Loss_G = Loss_recon
        Loss_G = Loss_recon + self.args.lamda1*(Loss_trip_p + Loss_trip_s) + self.args.lamda2*Loss_cross_recon
        # Loss_G = Loss_recon + self.args.lamda*Loss_trip_p + (1-self.args.lamda)*Loss_trip_s + Loss_cross_recon

        # Loss_G = Loss_recon +  Loss_trip_p*(1./Loss_trip_p_number) + Loss_trip_s*(1./Loss_trip_s_number)
        # Loss_G = Loss_recon +  Loss_trip_p*(1./Loss_trip_p_number)

        # Loss_G = Loss_recon + Loss_cross_recon + 0.5 * Loss_trip_p + 0.5 * Loss_trip_s

        # global writter
        # writter.add_scalar('Loss_recon',Loss_recon,self.trainer.global_step)
        loss_dict = {
            'Loss_G': Loss_G,
            'loss_recon': Loss_recon,
            'Loss_cross_recon': Loss_cross_recon,
            'Loss_trip_p': Loss_trip_p,
            'Loss_trip_s': Loss_trip_s
        }
        self.log_dict(loss_dict, logger=False, prog_bar=True, on_step=True)
        # if self.global_step % 200 == 0:
        if self.trainer.current_epoch % 100 == 0:
            self.trainer.save_checkpoint(os.path.join(self.hparams.model_path, f"{self.trainer.current_epoch}_final_model.ckpt"))
            current_model = deepcopy(self.trainer.model)
            gen_latent(data_loader=self.trainer.datamodule.train_dataloader(), model=current_model.cpu(), save_dir=self.args.latent_path, role='train', index=self.trainer.current_epoch)
            # torch.save(self.trainer.state_dict(),os.path.join(args.model_path, f"{self.trainer.current_epoch}_model.pth"))
        return Loss_G

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        data1 = batch['x'].float().to(self.device)
        data1_recon, p1, s1 = self(data1)
        Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # global writter
        # writter.add_scalar("val_loss_recon",Loss_recon,self.trainer.global_step)
        self.log("val_loss_recon", Loss_recon, on_step=True, logger=False, prog_bar=True)

        # self.log("val_recon",Loss_recon,on_step=True,on_epoch=True, prog_bar=True, logger=True)
        # return {"loss":Loss_recon}

    def test_step(self, batch, batch_idx):
        data1 = batch['x'].float().to(self.device)
        p1_id_gt = batch["p"].long()
        s1_id_gt = batch["s"].long()
        data1_recon, p1, s1 = self(data1)
        # y = batch["p"].type_as(batch["s"]) - 1
        loss = self.loss["reconstruction_criterion"](data1_recon, data1)
        # Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # self.log("test_recon",loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        output = {"loss_recon": loss, "oridata": data1, "p1": p1, "s1": s1, "y_p": p1_id_gt, "y_s": s1_id_gt}
        self.test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        output = self.test_outputs
        loss_recon = []
        oridata = []
        p1 = []
        s1 = []
        y_p = []
        y_s = []
        # losses=0
        for single_out in output:
            loss_recon.append(single_out["loss_recon"])
            oridata.append(single_out["oridata"])
            p1.append(single_out["p1"])
            s1.append(single_out["s1"])
            y_p.append(single_out["y_p"])
            y_s.append(single_out["y_s"])
        loss_recon = torch.stack(loss_recon).mean()
        
        # 绘图
        self.draw_tsne(torch.cat(oridata, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(p1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(s1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy(),
                torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy())
        
        oridata = torch.cat(oridata, dim=0).squeeze(0).detach().cpu().numpy()
        p1 = torch.cat(p1, dim=0).squeeze(0).detach().cpu().numpy()
        s1 = torch.cat(s1, dim=0).squeeze(0).detach().cpu().numpy()
        y_p = torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy()
        y_s = torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy()
        np.savez(os.path.join(self.hparams.latent_path, "feature.npz"),
                 oridata=oridata,
                 p1=p1,
                 s1=s1,
                 y_p=y_p,
                 y_s=y_s,
                 loss_recon=loss_recon.detach().cpu().numpy())
        print("save done....")

        self.log("test_loss_recon", loss_recon, on_step=False, logger=False, prog_bar=True)
        self.test_outputs.clear()

    def draw_tsne(self,oridata,p1,s1,y_p,y_s):
        
        colorlist = [
                "#FF0000",  # Red
                "#00FF00",  # Lime
                "#0000FF",  # Blue
                "#FFFF00",  # Yellow
                "#FF00FF",  # Magenta
                "#00FFFF",  # Cyan
                "#FFA500",  # Orange
                "#800080",  # Purple
                "#FFC0CB",  # Pink
                "#008080",  # Teal
                "#008000",  # Green
                "#000080",  # Navy
                "#FFD700",  # Gold
                "#FF4500",  # Orange Red
                "#800000",  # Dark Red
                "#FF1493",  # Deep Pink
                "#FF8C00",  # Dark Orange
                "#000000"   # Black
            ]
        markerlist =["o","v","^","s","p","P","*","h","H","+","x","X","D","d",'.', ',', '_', '|',"<",">",'*']
        
        
        pattern_uni=np.unique(y_p)
        colors=np.random.choice(colorlist,len(pattern_uni),replace=False)
        colors_dict=dict(zip(pattern_uni,colors))

        subject_uni=np.unique(y_s)
        markers=np.random.choice(markerlist,len(subject_uni),replace=False)
        markers_dict=dict(zip(subject_uni,markers))

        # color = ['limegreen', 'cornflowerblue', 'orange']
        
        fig, ax = plt.subplots(nrows=1,ncols=3,dpi=300,figsize=(18,6))


        tsne = _fit_tsne(oridata)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[0].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]], alpha=0.7)
            ax[0].set_title('original data')

        tsne = _fit_tsne(p1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[1].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[1].set_title('pattern component')

        tsne = _fit_tsne(s1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[2].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[2].set_title('subject component')

        subject_handles = [plt.Line2D([], [], color='gray', linestyle='None',marker=markers_dict[subject], markersize=8) for subject in subject_uni]
        subject_labels = [subject for subject in subject_uni]
        # Create legend handles and labels for patterns
        pattern_handles = [plt.Line2D([], [], color=colors_dict[pattern],linestyle='None', marker='o', markersize=8) for pattern in pattern_uni]
        pattern_labels = [pattern for pattern in pattern_uni]
        # Add legend to the plot
        subject_legend = ax[2].legend(subject_handles, subject_labels, loc='upper right',bbox_to_anchor=(1.7, 0.65),title='Subjects',frameon=False,ncol=3)
        ax[2].add_artist(subject_legend)
        # Create the pattern legend section
        pattern_legend = ax[2].legend(pattern_handles, pattern_labels, loc='upper right',bbox_to_anchor=(1.7, 1),title='Patterns',frameon=False,ncol=3)
        # Add both legends to the plot
        ax[2].add_artist(pattern_legend)

        fig.subplots_adjust(right=0.8)

        fig.show()
        fig.savefig(os.path.join(self.hparams.output_path,f'tsne_{self.args.purpose}.jpg'))


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

class Generator_gan(pl.LightningModule):
    def __init__(self, args):
        super(Generator_gan, self).__init__()

        if isinstance(args, dict):
            args = Dict2Obj(args)

        self.args = args
        self.encoder_p = Encoder_p(channels=args.encoder_channels, kernel_size=4)
        self.encoder_s = Encoder_s(channels=args.encoder_channels, kernel_size=4)
        if args.decoder_type == 'vae':
            self.decoder = Decoder_vae(channels=[256, 128, 64, len(args.features)*args.feature_dim])
        elif args.decoder_type == 'orig':
            self.decoder = Decoder(channels=[256, 128, 64, len(args.features)*args.feature_dim])
        self.discriminator = Discriminator_gan(image_size=16, channels=len(args.features)*args.feature_dim)
        self.save_hyperparameters(args)

    def forward(self, x):
        p = self.encoder_p(x)
        # print("p shape",p.shape)
        s = self.encoder_s(x)
        # print("s shape", s.shape)
        # print("concat shape",torch.cat([p, s], dim=1).shape)
        x_recon = self.decoder(torch.cat([p, s], dim=1))

        return x_recon, p, s

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def cross(self, x_p1_sk, x_pk_s1):  # 交叉重构
        p1 = self.encoder_p(x_p1_sk)
        sk = self.encoder_s(x_p1_sk)
        pk = self.encoder_p(x_pk_s1)
        s1 = self.encoder_s(x_pk_s1)
        out_p1_s1 = self.decoder(torch.cat([p1, s1], dim=1))
        return out_p1_s1
    
    @property
    def loss(self):
        return {
            "reconstruction_criterion": nn.MSELoss(),
            "trip_criterion": nn.TripletMarginLoss(margin=0.3),
            "p_criterion": nn.CrossEntropyLoss(),
            "s_criterion": nn.CrossEntropyLoss(),
            "dis_criterion": nn.MSELoss()
        }

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(itertools.chain(self.encoder_p.parameters(), self.encoder_s.parameters(), self.decoder.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.scheduler_t_max, eta_min=0, last_epoch=-1, verbose=False)
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, gamma=0.5, step_size=200)
        dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, gamma=0.5, step_size=200)
        return [gen_optimizer,dis_optimizer], [gen_scheduler,dis_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        p1_id_gt = batch["p"].type(torch.LongTensor)
        s1_id_gt = batch["s"].type(torch.LongTensor)
        batch_size = data1.size(0)
        data1_recon, p1, s1 = self(data1)
        p1 = p1.reshape(p1.shape[0], -1)
        s1 = s1.reshape(s1.shape[0], -1)

        # convert t
        s1_id_gt = Variable(s1_id_gt.cuda())
        p1_id_gt = Variable(p1_id_gt.cuda())
        data1 = Variable(data1.cuda())

        ##  # train generator
        if optimizer_idx == 0:
            ## reconstruction loss
            Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)

            # # triplet loss pattern
            # # Loss_trip_p = 0
            Loss_trip_p = 0
            Loss_trip_p_number = 0
            achor_p=[]
            positive_p=[]
            negative_p=[]
            for i in np.unique(np.unique(p1_id_gt.data.cpu())):
                for j in np.unique(s1_id_gt.data.cpu()):
                    data_index_pisj = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)
                    data_index_pisk = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_pksl = ((p1_id_gt != i)).nonzero().squeeze(dim=-1)
                    # why dim need to greater than 0, what about [index]?
                    if data_index_pisj.dim() > 0:
                        data_len = min(len(data_index_pisj), len(data_index_pisk), len(data_index_pksl))
                        # print(f"find availibale pattern labels --> len:{data_len}")

                        if data_len == 0:
                            continue
                        data_index_pisk = data_index_pisk[0:data_len].squeeze(dim=-1)
                        data_index_pksl = data_index_pksl[0:data_len].squeeze(dim=-1)
                        data_index_pisj = data_index_pisj[0:data_len].squeeze(dim=-1)

                        # print('pisk:{:.0f} pksl:{:.0f} pisj:{:.0f}'
                        #      .format(len(p1[data_index_pisk]),len(p1[data_index_pksl]),len(data_index_pisj)))
                        if data_len < 2:
                            a = p1[data_index_pisj].unsqueeze(dim=0)
                            p = p1[data_index_pisk].unsqueeze(dim=0)
                            n = p1[data_index_pksl].unsqueeze(dim=0)
                        else:
                            a = p1[data_index_pisj]
                            p = p1[data_index_pisk]
                            n = p1[data_index_pksl]
                        achor_p.append(a)
                        positive_p.append(p)
                        negative_p.append(n)
                        # Loss_trip_p_number += 1
                    else:
                        continue
            if len(achor_p) == 0 or len(positive_p)==0 or len(negative_p)==0:
                return None
            achor_p=torch.cat(achor_p,dim=0)
            positive_p=torch.cat(positive_p,dim=0)
            negative_p=torch.cat(negative_p,dim=0)
            Loss_trip_p = (self.loss["trip_criterion"](achor_p, positive_p, negative_p))

            # # Triplet loss subjects
            # # Loss_trip_s = 0
            achor_s=[]
            positive_s=[]
            negative_s=[]
            Loss_trip_s = 0
            Loss_trip_s_number = 0
            for i in np.unique(s1_id_gt.data.cpu()):
                for j in np.unique(p1_id_gt.data.cpu()):
                    data_index_sipj = ((s1_id_gt == i) * (p1_id_gt == j)).nonzero().squeeze(dim=-1)
                    data_index_sipk = ((s1_id_gt == i) * (p1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_skpl = ((s1_id_gt != i)).nonzero().squeeze(dim=-1)

                    if data_index_sipj.dim() > 0:
                        data_len = min(len(data_index_sipj), len(data_index_sipk), len(data_index_skpl))
                        # print(f"find availibale subject labels --> len:{data_len}")
                        if data_len == 0:
                            continue
                        data_index_sipk = data_index_sipk[0:data_len].squeeze(dim=-1)
                        data_index_skpl = data_index_skpl[0:data_len].squeeze(dim=-1)
                        data_index_sipj = data_index_sipj[0:data_len].squeeze(dim=-1)
                        # print('sipk:{:.0f} skpl:{:.0f} sipj:{:.0f}'
                        #      .format(len(s1[data_index_sipk]),len(s1[data_index_skpl]),len(data_index_sipj)))
                        # print(data_len)
                        if data_len < 2:
                            a = s1[data_index_sipj].unsqueeze(dim=0)
                            p = s1[data_index_sipk].unsqueeze(dim=0)
                            n = s1[data_index_skpl].unsqueeze(dim=0)
                        else:
                            a = s1[data_index_sipj]
                            p = s1[data_index_sipk]
                            n = s1[data_index_skpl]
                        achor_s.append(a)
                        positive_s.append(p)
                        negative_s.append(n)
                        # Loss_trip_s += (self.loss["trip_criterion"](a, p, n))
                        # print("loss trip s:",Loss_trip_s)
                        Loss_trip_s_number += 1
                        # validate_loss += data_len.
                    else:
                        continue
            if len(achor_s) == 0 or len(positive_s)==0 or len(negative_s)==0:
                return None
            achor_s=torch.cat(achor_s,dim=0)
            positive_s=torch.cat(positive_s,dim=0)
            negative_s=torch.cat(negative_s,dim=0)
            Loss_trip_s = (self.loss["trip_criterion"](achor_s, positive_s, negative_s))

            # cross_reconstruction
            Loss_cross_recon = 0
            Loss_cross_recon_number = 0
            pi=[]
            sj=[]
            pisj=[]
            for i in np.unique(p1_id_gt.data.cpu()):
                for j in np.unique(s1_id_gt.data.cpu()):
                    # data_index = (p1_id_gt==i) and (s1_id_gt==j)
                    data_index_p = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_s = ((s1_id_gt == j) * (p1_id_gt != i)).nonzero().squeeze(dim=-1)
                    data_index = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)

                    if data_index.dim() > 0:

                        # data_index = random.shuffle(data_index)
                        data_index_p = data_index_p[torch.randperm(len(data_index_p))]
                        data_index_s = data_index_s[torch.randperm(len(data_index_s))]

                        data_len = min(len(data_index), len(data_index_p), len(data_index_s))
                        if data_len < 2:
                            continue
                        data_index_p = data_index_p[0:data_len]
                        data_index_s = data_index_s[0:data_len]
                        data_index = data_index[0:data_len]

                        pi.append(data1[data_index_p])
                        sj.append(data1[data_index_s])
                        pisj.append(data1[data_index])
                        # print('p:{:.0f} s:{:.0f} ps:{:.0f}'
                        #      .format(len(data1[data_index_p]),len(data1[data_index_s]),len(data_index)))
                    else:
                        continue
            if len(pi) == 0 or len(sj)==0 or len(pisj)==0:
                return None
            pi=torch.cat(pi,dim=0)
            sj=torch.cat(sj,dim=0)
            pisj=torch.cat(pisj,dim=0)
            outpisj = self.cross(pi, sj)
            Loss_cross_recon=self.loss["reconstruction_criterion"](outpisj, pisj)

            Loss_G = Loss_recon + 0.5*Loss_trip_p + 0.5*Loss_trip_s + Loss_cross_recon

            # global writter
            # writter.add_scalar('Loss_recon',Loss_recon,self.trainer.global_step)
            loss_dict = {
                'Loss_G': Loss_G,
                'Loss_recon': Loss_recon,
                'Loss_cross_recon': Loss_cross_recon,
                'Loss_trip_p': Loss_trip_p,
                'Loss_trip_s': Loss_trip_s
            }
            self.log_dict(loss_dict, logger=False, prog_bar=True, on_step=True)
            # if self.global_step % 200 == 0:
            if self.trainer.current_epoch % 100 == 0:
                # self.train_dataloader
                self.trainer.save_checkpoint(os.path.join(self.hparams.model_path, f"{self.trainer.current_epoch}_final_model.ckpt"))
                current_model = deepcopy(self.trainer.model)
                gen_latent(data_loader=self.trainer.datamodule.train_dataloader(), model=current_model.cpu(), save_dir=self.args.latent_path, role='train', index=self.trainer.current_epoch)
                # torch.save(self.trainer.state_dict(),os.path.join(args.model_path, f"{self.trainer.current_epoch}_model.pth"))
            return Loss_G
        
        ##  # train discriminator
        if optimizer_idx == 1:
            ## discriminator loss
            real_label = torch.full((batch_size, 1), 1, dtype=data1.dtype).cuda()
            fake_label = torch.full((batch_size, 1), 0, dtype=data1.dtype).cuda()
            real_loss = self.adversarial_loss(self.discriminator(data1), real_label)
            fake_loss = self.adversarial_loss(self.discriminator(data1_recon), fake_label)
            Loss_D = 0.5*real_loss + 0.5*fake_loss
            # loss_dict = {
            #     'Loss_D': Loss_D,
            # }
            # # self.log_dict(loss_dict,logger=True,prog_bar=True,on_step=True)
            # self.log("Train", loss_dict, logger=True, prog_bar=True, on_step=True)
            # self.log("Loss_D", Loss_D, logger=True, prog_bar=True, on_step=True)
            return Loss_D

    def training_epoch_end(self, outputs):
        print(outputs)

    def validation_step(self, batch, batch_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        data1_recon, p1, s1 = self(data1)
        Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # global writter
        # writter.add_scalar("val_loss_recon",Loss_recon,self.trainer.global_step)
        self.log("val_loss_recon", Loss_recon, on_step=True, logger=False, prog_bar=True)

        # self.log("val_recon",Loss_recon,on_step=True,on_epoch=True, prog_bar=True, logger=True)
        # return {"loss":Loss_recon}

    def test_step(self, batch, batch_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        p1_id_gt = batch["p"].type(torch.LongTensor)
        s1_id_gt = batch["s"].type(torch.LongTensor)
        data1_recon, p1, s1 = self(data1)
        # y = batch["p"].type_as(batch["s"]) - 1
        loss = self.loss["reconstruction_criterion"](data1_recon, data1)
        # Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # self.log("test_recon",loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        return {"loss_recon": loss, "oridata": data1, "p1": p1, "s1": s1, "y_p": p1_id_gt, "y_s": s1_id_gt}

    def test_epoch_end(self, output):
        loss_recon = []
        oridata = []
        p1 = []
        s1 = []
        y_p = []
        y_s = []
        # losses=0
        for single_out in output:
            loss_recon.append(single_out["loss_recon"])
            oridata.append(single_out["oridata"])
            p1.append(single_out["p1"])
            s1.append(single_out["s1"])
            y_p.append(single_out["y_p"])
            y_s.append(single_out["y_s"])
        loss_recon = torch.stack(loss_recon).mean()
        
        # 绘图
        self.draw_tsne(torch.cat(oridata, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(p1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(s1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy(),
                torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy())
        
        oridata = torch.cat(oridata, dim=0).squeeze(0).detach().cpu().numpy()
        p1 = torch.cat(p1, dim=0).squeeze(0).detach().cpu().numpy()
        s1 = torch.cat(s1, dim=0).squeeze(0).detach().cpu().numpy()
        y_p = torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy()
        y_s = torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy()
        np.savez(os.path.join(self.hparams.latent_path, "feature.npz"),
                 oridata=oridata,
                 p1=p1,
                 s1=s1,
                 y_p=y_p,
                 y_s=y_s,
                 loss_recon=loss_recon.detach().cpu().numpy())
        print("save done....")

        self.log("test_loss_recon", loss_recon, on_step=False, logger=False, prog_bar=True)

    def draw_tsne(self,oridata,p1,s1,y_p,y_s):
        
        colorlist = [
                "#FF0000",  # Red
                "#00FF00",  # Lime
                "#0000FF",  # Blue
                "#FFFF00",  # Yellow
                "#FF00FF",  # Magenta
                "#00FFFF",  # Cyan
                "#FFA500",  # Orange
                "#800080",  # Purple
                "#FFC0CB",  # Pink
                "#008080",  # Teal
                "#008000",  # Green
                "#000080",  # Navy
                "#FFD700",  # Gold
                "#FF4500",  # Orange Red
                "#800000",  # Dark Red
                "#FF1493",  # Deep Pink
                "#FF8C00",  # Dark Orange
                "#000000"   # Black
            ]
        markerlist =["o","v","^","s","p","P","*","h","H","+","x","X","D","d",'.', ',', '_', '|',"<",">",'*']
        
        
        pattern_uni=np.unique(y_p)
        colors=np.random.choice(colorlist,len(pattern_uni),replace=False)
        colors_dict=dict(zip(pattern_uni,colors))

        subject_uni=np.unique(y_s)
        markers=np.random.choice(markerlist,len(subject_uni),replace=False)
        markers_dict=dict(zip(subject_uni,markers))

        # color = ['limegreen', 'cornflowerblue', 'orange']
        
        fig, ax = plt.subplots(nrows=1,ncols=3,dpi=300,figsize=(18,6))


        tsne = _fit_tsne(oridata)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[0].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]], alpha=0.7)
            ax[0].set_title('original data')

        tsne = _fit_tsne(p1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[1].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[1].set_title('pattern component')

        tsne = _fit_tsne(s1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[2].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[2].set_title('subject component')

        subject_handles = [plt.Line2D([], [], color='gray', linestyle='None',marker=markers_dict[subject], markersize=8) for subject in subject_uni]
        subject_labels = [subject for subject in subject_uni]
        # Create legend handles and labels for patterns
        pattern_handles = [plt.Line2D([], [], color=colors_dict[pattern],linestyle='None', marker='o', markersize=8) for pattern in pattern_uni]
        pattern_labels = [pattern for pattern in pattern_uni]
        # Add legend to the plot
        subject_legend = ax[2].legend(subject_handles, subject_labels, loc='upper right',bbox_to_anchor=(1.7, 0.65),title='Subjects',frameon=False,ncol=3)
        ax[2].add_artist(subject_legend)
        # Create the pattern legend section
        pattern_legend = ax[2].legend(pattern_handles, pattern_labels, loc='upper right',bbox_to_anchor=(1.7, 1),title='Patterns',frameon=False,ncol=3)
        # Add both legends to the plot
        ax[2].add_artist(pattern_legend)

        fig.subplots_adjust(right=0.8)

        fig.show()
        fig.savefig(os.path.join(self.hparams.output_path,f'tsne_{self.args.purpose}.jpg'))


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

class Generator_multiinput(pl.LightningModule):
    def __init__(self, args):
        super(Generator_multiinput, self).__init__()

        if isinstance(args, dict):
            args = Dict2Obj(args)

        self.args = args
        self.encoder_p_stft = Encoder_p(channels=[256, 512, 256, 128], kernel_size=4)
        self.encoder_s_stft = Encoder_s(channels=[256, 512, 256, 128], kernel_size=4)
        self.encoder_p_fea = Encoder_p(channels=[4, 512, 256, 128], kernel_size=4)
        self.encoder_s_fea = Encoder_s(channels=[4, 512, 256, 128], kernel_size=4)
        if args.decoder_type == 'vae':
            self.decoder = Decoder_vae(channels=[512, 128, 64, len(args.features)*args.feature_dim])
        elif args.decoder_type == 'orig':
            self.decoder = Decoder(channels=[512, 128, 64, len(args.features)*args.feature_dim])
        self.discriminator = Discriminator_gan(image_size=16, channels=len(args.features)*args.feature_dim)
        self.save_hyperparameters(args)

    def forward(self, x):
        p_stft = self.encoder_p_stft(x[:, :256, :, :])
        s_stft = self.encoder_s_stft(x[:, :256, :, :])
        p_fea = self.encoder_p_fea(x[:, 256:, :, :])
        s_fea = self.encoder_s_fea(x[:, 256:, :, :])
        p = torch.cat([p_stft, p_fea], dim=1)
        s = torch.cat([s_stft, s_fea], dim=1)
        x_recon = self.decoder(torch.cat([p, s], dim=1))

        return x_recon, p, s

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def cross(self, x_p1_sk, x_pk_s1):  # 交叉重构
        p1 = torch.cat([self.encoder_p_stft(x_p1_sk[:,:256,:,:]), self.encoder_p_fea(x_p1_sk[:,256:,:,:])], dim=1)
        sk = torch.cat([self.encoder_s_stft(x_p1_sk[:,:256,:,:]), self.encoder_s_fea(x_p1_sk[:,256:,:,:])], dim=1)
        pk = torch.cat([self.encoder_p_stft(x_pk_s1[:,:256,:,:]), self.encoder_p_fea(x_pk_s1[:,256:,:,:])], dim=1)
        s1 = torch.cat([self.encoder_s_stft(x_pk_s1[:,:256,:,:]), self.encoder_s_fea(x_pk_s1[:,256:,:,:])], dim=1)
        out_p1_s1 = self.decoder(torch.cat([p1, s1], dim=1))
        return out_p1_s1
    
    @property
    def loss(self):
        return {
            "reconstruction_criterion": nn.MSELoss(),
            "trip_criterion": nn.TripletMarginLoss(margin=0.3),
            "p_criterion": nn.CrossEntropyLoss(),
            "s_criterion": nn.CrossEntropyLoss(),
            "dis_criterion": nn.MSELoss()
        }

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(itertools.chain(self.encoder_p_stft.parameters(), self.encoder_p_fea.parameters(), self.encoder_s_stft.parameters(), self.encoder_s_fea.parameters(), self.decoder.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.scheduler_t_max, eta_min=0, last_epoch=-1, verbose=False)
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optimizer, gamma=0.5, step_size=200)
        dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, gamma=0.5, step_size=200)
        return [gen_optimizer,dis_optimizer], [gen_scheduler,dis_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        p1_id_gt = batch["p"].type(torch.LongTensor)
        s1_id_gt = batch["s"].type(torch.LongTensor)
        batch_size = data1.size(0)
        data1_recon, p1, s1 = self(data1)
        p1 = p1.reshape(p1.shape[0], -1)
        s1 = s1.reshape(s1.shape[0], -1)

        # convert t
        s1_id_gt = Variable(s1_id_gt.cuda())
        p1_id_gt = Variable(p1_id_gt.cuda())
        data1 = Variable(data1.cuda())

        ##  # train generator
        if optimizer_idx == 0:
            ## reconstruction loss
            Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)

            # # triplet loss pattern
            # # Loss_trip_p = 0
            Loss_trip_p = 0
            Loss_trip_p_number = 0
            achor_p=[]
            positive_p=[]
            negative_p=[]
            for i in np.unique(np.unique(p1_id_gt.data.cpu())):
                for j in np.unique(s1_id_gt.data.cpu()):
                    data_index_pisj = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)
                    data_index_pisk = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_pksl = ((p1_id_gt != i)).nonzero().squeeze(dim=-1)
                    # why dim need to greater than 0, what about [index]?
                    if data_index_pisj.dim() > 0:
                        data_len = min(len(data_index_pisj), len(data_index_pisk), len(data_index_pksl))
                        # print(f"find availibale pattern labels --> len:{data_len}")

                        if data_len == 0:
                            continue
                        data_index_pisk = data_index_pisk[0:data_len].squeeze(dim=-1)
                        data_index_pksl = data_index_pksl[0:data_len].squeeze(dim=-1)
                        data_index_pisj = data_index_pisj[0:data_len].squeeze(dim=-1)

                        # print('pisk:{:.0f} pksl:{:.0f} pisj:{:.0f}'
                        #      .format(len(p1[data_index_pisk]),len(p1[data_index_pksl]),len(data_index_pisj)))
                        if data_len < 2:
                            a = p1[data_index_pisj].unsqueeze(dim=0)
                            p = p1[data_index_pisk].unsqueeze(dim=0)
                            n = p1[data_index_pksl].unsqueeze(dim=0)
                        else:
                            a = p1[data_index_pisj]
                            p = p1[data_index_pisk]
                            n = p1[data_index_pksl]
                        achor_p.append(a)
                        positive_p.append(p)
                        negative_p.append(n)
                        # Loss_trip_p_number += 1
                    else:
                        continue
            if len(achor_p) == 0 or len(positive_p)==0 or len(negative_p)==0:
                return None
            achor_p=torch.cat(achor_p,dim=0)
            positive_p=torch.cat(positive_p,dim=0)
            negative_p=torch.cat(negative_p,dim=0)
            Loss_trip_p = (self.loss["trip_criterion"](achor_p, positive_p, negative_p))

            # # Triplet loss subjects
            # # Loss_trip_s = 0
            achor_s=[]
            positive_s=[]
            negative_s=[]
            Loss_trip_s = 0
            Loss_trip_s_number = 0
            for i in np.unique(s1_id_gt.data.cpu()):
                for j in np.unique(p1_id_gt.data.cpu()):
                    data_index_sipj = ((s1_id_gt == i) * (p1_id_gt == j)).nonzero().squeeze(dim=-1)
                    data_index_sipk = ((s1_id_gt == i) * (p1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_skpl = ((s1_id_gt != i)).nonzero().squeeze(dim=-1)

                    if data_index_sipj.dim() > 0:
                        data_len = min(len(data_index_sipj), len(data_index_sipk), len(data_index_skpl))
                        # print(f"find availibale subject labels --> len:{data_len}")
                        if data_len == 0:
                            continue
                        data_index_sipk = data_index_sipk[0:data_len].squeeze(dim=-1)
                        data_index_skpl = data_index_skpl[0:data_len].squeeze(dim=-1)
                        data_index_sipj = data_index_sipj[0:data_len].squeeze(dim=-1)
                        # print('sipk:{:.0f} skpl:{:.0f} sipj:{:.0f}'
                        #      .format(len(s1[data_index_sipk]),len(s1[data_index_skpl]),len(data_index_sipj)))
                        # print(data_len)
                        if data_len < 2:
                            a = s1[data_index_sipj].unsqueeze(dim=0)
                            p = s1[data_index_sipk].unsqueeze(dim=0)
                            n = s1[data_index_skpl].unsqueeze(dim=0)
                        else:
                            a = s1[data_index_sipj]
                            p = s1[data_index_sipk]
                            n = s1[data_index_skpl]
                        achor_s.append(a)
                        positive_s.append(p)
                        negative_s.append(n)
                        # Loss_trip_s += (self.loss["trip_criterion"](a, p, n))
                        # print("loss trip s:",Loss_trip_s)
                        Loss_trip_s_number += 1
                        # validate_loss += data_len.
                    else:
                        continue
            if len(achor_s) == 0 or len(positive_s)==0 or len(negative_s)==0:
                return None
            achor_s=torch.cat(achor_s,dim=0)
            positive_s=torch.cat(positive_s,dim=0)
            negative_s=torch.cat(negative_s,dim=0)
            Loss_trip_s = (self.loss["trip_criterion"](achor_s, positive_s, negative_s))

            # cross_reconstruction
            Loss_cross_recon = 0
            Loss_cross_recon_number = 0
            pi=[]
            sj=[]
            pisj=[]
            for i in np.unique(p1_id_gt.data.cpu()):
                for j in np.unique(s1_id_gt.data.cpu()):
                    # data_index = (p1_id_gt==i) and (s1_id_gt==j)
                    data_index_p = ((p1_id_gt == i) * (s1_id_gt != j)).nonzero().squeeze(dim=-1)
                    data_index_s = ((s1_id_gt == j) * (p1_id_gt != i)).nonzero().squeeze(dim=-1)
                    data_index = ((p1_id_gt == i) * (s1_id_gt == j)).nonzero().squeeze(dim=-1)

                    if data_index.dim() > 0:

                        # data_index = random.shuffle(data_index)
                        data_index_p = data_index_p[torch.randperm(len(data_index_p))]
                        data_index_s = data_index_s[torch.randperm(len(data_index_s))]

                        data_len = min(len(data_index), len(data_index_p), len(data_index_s))
                        if data_len < 2:
                            continue
                        data_index_p = data_index_p[0:data_len]
                        data_index_s = data_index_s[0:data_len]
                        data_index = data_index[0:data_len]

                        pi.append(data1[data_index_p])
                        sj.append(data1[data_index_s])
                        pisj.append(data1[data_index])
                        # print('p:{:.0f} s:{:.0f} ps:{:.0f}'
                        #      .format(len(data1[data_index_p]),len(data1[data_index_s]),len(data_index)))
                    else:
                        continue
            if len(pi) == 0 or len(sj)==0 or len(pisj)==0:
                return None
            pi=torch.cat(pi,dim=0)
            sj=torch.cat(sj,dim=0)
            pisj=torch.cat(pisj,dim=0)
            outpisj = self.cross(pi, sj)
            Loss_cross_recon=self.loss["reconstruction_criterion"](outpisj, pisj)

            Loss_G = Loss_recon + 0.5*Loss_trip_p + 0.5*Loss_trip_s + Loss_cross_recon

            # global writter
            # writter.add_scalar('Loss_recon',Loss_recon,self.trainer.global_step)
            loss_dict = {
                'Loss_G': Loss_G,
                'Loss_recon': Loss_recon,
                'Loss_cross_recon': Loss_cross_recon,
                'Loss_trip_p': Loss_trip_p,
                'Loss_trip_s': Loss_trip_s
            }
            self.log_dict(loss_dict, logger=False, prog_bar=True, on_step=True)
            # if self.global_step % 200 == 0:
            if self.trainer.current_epoch % 100 == 0:
                # self.train_dataloader
                self.trainer.save_checkpoint(os.path.join(self.hparams.model_path, f"{self.trainer.current_epoch}_final_model.ckpt"))
                current_model = deepcopy(self.trainer.model)
                # gen_latent(data_loader=self.trainer.datamodule.train_dataloader(), model=current_model.cpu(), save_dir=self.args.latent_path, role='train', index=self.trainer.current_epoch)
                gen_latent_multi(data_loader=self.trainer.datamodule.train_dataloader(), model=current_model.cpu(), save_dir=self.args.latent_path, role='train', index=self.trainer.current_epoch)
                # torch.save(self.trainer.state_dict(),os.path.join(args.model_path, f"{self.trainer.current_epoch}_model.pth"))
            return Loss_G
        
        ##  # train discriminator
        if optimizer_idx == 1:
            ## discriminator loss
            real_label = torch.full((batch_size, 1), 1, dtype=data1.dtype).cuda()
            fake_label = torch.full((batch_size, 1), 0, dtype=data1.dtype).cuda()
            real_loss = self.adversarial_loss(self.discriminator(data1), real_label)
            fake_loss = self.adversarial_loss(self.discriminator(data1_recon), fake_label)
            Loss_D = 0.5*real_loss + 0.5*fake_loss
            # loss_dict = {
            #     'Loss_D': Loss_D,
            # }
            # # self.log_dict(loss_dict,logger=True,prog_bar=True,on_step=True)
            # self.log("Train", loss_dict, logger=True, prog_bar=True, on_step=True)
            # self.log("Loss_D", Loss_D, logger=True, prog_bar=True, on_step=True)
            return Loss_D

    def training_epoch_end(self, outputs):
        print(outputs)

    def validation_step(self, batch, batch_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        data1_recon, p1, s1 = self(data1)
        Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # global writter
        # writter.add_scalar("val_loss_recon",Loss_recon,self.trainer.global_step)
        self.log("val_loss_recon", Loss_recon, on_step=True, logger=False, prog_bar=True)

        # self.log("val_recon",Loss_recon,on_step=True,on_epoch=True, prog_bar=True, logger=True)
        # return {"loss":Loss_recon}

    def test_step(self, batch, batch_idx):
        data1 = batch['x'].type(torch.cuda.FloatTensor)
        p1_id_gt = batch["p"].type(torch.LongTensor)
        s1_id_gt = batch["s"].type(torch.LongTensor)
        data1_recon, p1, s1 = self(data1)
        # y = batch["p"].type_as(batch["s"]) - 1
        loss = self.loss["reconstruction_criterion"](data1_recon, data1)
        # Loss_recon = self.loss["reconstruction_criterion"](data1_recon, data1)
        # self.log("test_recon",loss,on_step=False,on_epoch=True, prog_bar=True, logger=True)
        return {"loss_recon": loss, "oridata": data1, "p1": p1, "s1": s1, "y_p": p1_id_gt, "y_s": s1_id_gt}

    def test_epoch_end(self, output):
        loss_recon = []
        oridata = []
        p1 = []
        s1 = []
        y_p = []
        y_s = []
        # losses=0
        for single_out in output:
            loss_recon.append(single_out["loss_recon"])
            oridata.append(single_out["oridata"])
            p1.append(single_out["p1"])
            s1.append(single_out["s1"])
            y_p.append(single_out["y_p"])
            y_s.append(single_out["y_s"])
        loss_recon = torch.stack(loss_recon).mean()
        
        # 绘图
        self.draw_tsne(torch.cat(oridata, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(p1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(s1, dim=0).squeeze(0).flatten(1).detach().cpu().numpy(),
                torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy(),
                torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy())
        
        oridata = torch.cat(oridata, dim=0).squeeze(0).detach().cpu().numpy()
        p1 = torch.cat(p1, dim=0).squeeze(0).detach().cpu().numpy()
        s1 = torch.cat(s1, dim=0).squeeze(0).detach().cpu().numpy()
        y_p = torch.cat(y_p, dim=0).squeeze(0).detach().cpu().numpy()
        y_s = torch.cat(y_s, dim=0).squeeze(0).detach().cpu().numpy()
        np.savez(os.path.join(self.hparams.latent_path, "feature.npz"),
                 oridata=oridata,
                 p1=p1,
                 s1=s1,
                 y_p=y_p,
                 y_s=y_s,
                 loss_recon=loss_recon.detach().cpu().numpy())
        print("save done....")

        self.log("test_loss_recon", loss_recon, on_step=False, logger=False, prog_bar=True)

    def draw_tsne(self,oridata,p1,s1,y_p,y_s):
        
        colorlist = [
                "#FF0000",  # Red
                "#00FF00",  # Lime
                "#0000FF",  # Blue
                "#FFFF00",  # Yellow
                "#FF00FF",  # Magenta
                "#00FFFF",  # Cyan
                "#FFA500",  # Orange
                "#800080",  # Purple
                "#FFC0CB",  # Pink
                "#008080",  # Teal
                "#008000",  # Green
                "#000080",  # Navy
                "#FFD700",  # Gold
                "#FF4500",  # Orange Red
                "#800000",  # Dark Red
                "#FF1493",  # Deep Pink
                "#FF8C00",  # Dark Orange
                "#000000"   # Black
            ]
        markerlist =["o","v","^","s","p","P","*","h","H","+","x","X","D","d",'.', ',', '_', '|',"<",">",'*']
        
        
        pattern_uni=np.unique(y_p)
        colors=np.random.choice(colorlist,len(pattern_uni),replace=False)
        colors_dict=dict(zip(pattern_uni,colors))

        subject_uni=np.unique(y_s)
        markers=np.random.choice(markerlist,len(subject_uni),replace=False)
        markers_dict=dict(zip(subject_uni,markers))

        # color = ['limegreen', 'cornflowerblue', 'orange']
        
        fig, ax = plt.subplots(nrows=1,ncols=3,dpi=300,figsize=(18,6))


        tsne = _fit_tsne(oridata)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[0].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]], alpha=0.7)
            ax[0].set_title('original data')

        tsne = _fit_tsne(p1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[1].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[1].set_title('pattern component')

        tsne = _fit_tsne(s1)
        tsne_x = tsne[:, 0]
        tsne_y = tsne[:, 1]
        # ax.figure(dpi=300)
        for i in range(tsne.shape[0]):
            ax[2].scatter(tsne_x[i], tsne_y[i], facecolor=colors_dict[y_p[i]], marker=markers_dict[y_s[i]],alpha=0.7)
            ax[2].set_title('subject component')

        subject_handles = [plt.Line2D([], [], color='gray', linestyle='None',marker=markers_dict[subject], markersize=8) for subject in subject_uni]
        subject_labels = [subject for subject in subject_uni]
        # Create legend handles and labels for patterns
        pattern_handles = [plt.Line2D([], [], color=colors_dict[pattern],linestyle='None', marker='o', markersize=8) for pattern in pattern_uni]
        pattern_labels = [pattern for pattern in pattern_uni]
        # Add legend to the plot
        subject_legend = ax[2].legend(subject_handles, subject_labels, loc='upper right',bbox_to_anchor=(1.7, 0.65),title='Subjects',frameon=False,ncol=3)
        ax[2].add_artist(subject_legend)
        # Create the pattern legend section
        pattern_legend = ax[2].legend(pattern_handles, pattern_labels, loc='upper right',bbox_to_anchor=(1.7, 1),title='Patterns',frameon=False,ncol=3)
        # Add both legends to the plot
        ax[2].add_artist(pattern_legend)

        fig.subplots_adjust(right=0.8)

        fig.show()
        fig.savefig(os.path.join(self.hparams.output_path,f'tsne_{self.args.purpose}.jpg'))


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


# classifier  
class ANNmodel(nn.Module):
    def __init__(self, feature_dimension, hidden_dim, num_class):
        super(ANNmodel, self).__init__()
		#定义层
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dimension, hidden_dim),
            nn.ReLU())  # nn.Linear为线性关系，加上激活函数转为非线性
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        
        self.fc3 = nn.Linear(hidden_dim, num_class)
               
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class CNNmodel(nn.Module):
    def __init__(self, feature_dimension, num_class, dropout_p=0.1):
        super(CNNmodel, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(feature_dimension, 32, kernel_size=2, stride=1,padding=0),
            nn.BatchNorm2d(32),
            # nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.LeakyReLU(.2),
            nn.ReLU(inplace=False),
            
            nn.Conv2d(32, 64, kernel_size=2, stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.LeakyReLU(.2),
            nn.ReLU(inplace=False),
        )
        self.fc=nn.Sequential(
            nn.Flatten(-3, -1),
            # nn.Linear(64 * 3 * 3, 256),
            
            nn.BatchNorm1d(3136),
            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            # nn.LeakyReLU(.2),
        )
        self.classifier_layer=nn.Linear(256, num_class)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        output = self.classifier_layer(x)
        return output
    

class LSTMmodel(nn.Module):
    def __init__(self, feature_dimension, num_class):
        super(LSTMmodel, self).__init__()
        # self.modal1_conv = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2,2), padding=(1, 1), bias=False)
        # self.modal1_conv2 = nn.Conv2d(465, 465, kernel_size=(1, 3), stride=(1,2), padding=(0, 0), bias=False)
        # 其余部分与之前的定义相同
        self.lstm = nn.LSTM(input_size=feature_dimension,hidden_size=48,num_layers=1,batch_first=True,dropout=0.5,bidirectional=False)
        self.maxpool=nn.MaxPool1d(16,stride=16)
        self.endclassifier=nn.Sequential(
        nn.Linear(6, 1024),  # nn.Linear(3072, 3072),
        # nn.BatchNorm1d(1536),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.4),
        nn.Linear(1024,num_class)
        )

    def forward(self, x):
        # x = self.modal1_conv(x)
        x=x.view(x.size(0),x.size(1),-1)
        # x=x.permute(0,2,1)  # lstm input shape: (sampls,batch,feature_dim) if batch_first==False
        lstmoutput,(_,_) = self.lstm(x)
        # lstmoutput=self.maxpool(lstmoutput.permute(0,2,1))
        # lstmoutput=lstmoutput.permute(0,2,1)  # 恢复原来维度
        lstmoutput=self.maxpool(lstmoutput)
        
        # lstmoutput =  torch.topk(lstmoutput, k=10, dim=1).values
        x=lstmoutput.flatten(1)
        # x =x[:,-1,:]
        x=self.endclassifier(x)
        # x=torch.softmax(x,1)
        return x

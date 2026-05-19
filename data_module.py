# normalize data by spatial
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import glob
import numpy as np
import os
import itertools
import torch
import re
from sklearn.preprocessing import StandardScaler, normalize
import scipy.io as scio
import matplotlib.pyplot as plt
# from data_preprocess.normalize_sample import normalize_samples

class Subdataset(Dataset):
    def __init__(self,samples):
        self.files=[]
        self.samples=samples
    def __getitem__(self,idx):
        return {
            "x":self.samples["x"][idx],
            "p":self.samples["p"][idx],
            "s":self.samples["s"][idx]
        }
    def __len__(self):
        return len(self.samples["p"])

def _load_data(sample_dict, file_dir, subj_id, selected_features):
    selected_patterns = [6, 7, 8, 9, 10, 11, 30, 31, 32, 34] 
    features = scio.loadmat(f'{file_dir}/pr_feature_smooth_dynamic.mat')['feature_smooth'][0]
    labels = scio.loadmat(f'{file_dir}/label_dynamic.mat')['label'][0]
    print(features.shape)

    # select patterns
    is_valid=np.isin(labels,selected_patterns)
    label=labels[is_valid]
    feature=features[is_valid]

    # select features
    feature_selected = []
    for samp in feature:
        sampfeature = []
        for feaidx in selected_features:
            tempfea = samp[0,feaidx][0]
            sampfeature.append(tempfea)
        sampfeature = np.concatenate(sampfeature,axis=0)[np.newaxis,:]
        feature_selected.append(sampfeature)
    feature_selected = np.concatenate(feature_selected,axis=0)    

    ## reshape
    feature=feature_selected
    # transform label to start from 0- 
    label=[selected_patterns.index(c) for c in label]
    sample_dict["x"].append(feature)
    sample_dict["p"].append(label)
    n_samples=len(label)
    sample_dict["s"].append(np.ones(n_samples, dtype=int) * (subj_id - 1))
    return sample_dict

def _preproess_dict(sample_dict):
    sample_dict["x"]=np.vstack(sample_dict["x"])
    sample_dict["p"]=np.hstack(sample_dict["p"])
    sample_dict["s"] = np.hstack(sample_dict["s"])
    # sample_dict["x"]=preprocessing.scale(sample_dict["x"],axis=0)
    return sample_dict


class DataModule(LightningDataModule):
    def __init__(self, batch_size=64, data_dir=None, features=[], feature_dim=1, test_id=0, opts=None, session_id=1, mode="C", purpose="train", shuffle=True):
        super().__init__()
        self.opts=opts
        self.batch_size=batch_size
        self.data_dir=data_dir
        self.test_id=test_id
        self.val_ratio=0
        self.num_workers=getattr(opts, "num_workers", 8)
        self.session_id=session_id
        self.mode=mode
        self.purpose=purpose
        self.train_samples = {
            "x": [],
            "p": [],
            "s": []
        }
        self.test_samples = {
            "x": [],
            "p": [],
            "s": []
        }
        self.features = features
        self.feature_dim = feature_dim
        self.shuffle = shuffle
        # self.opts.labels = np.array([1, 2, 6, 7, 13, 28, 30, 31, 32, 33]) - 1 if opts.class_num==10 else np.arange(opts.class_num)
    
    def prepare_data(self):
        filepath = self.data_dir
        subject_id_list = np.arange(1, 21)
        test_id_list = self.test_id
        train_id_list = list(set(subject_id_list) - set(test_id_list))

        for subj_id in train_id_list:
            print(f'load train dataset {subj_id}')
            file_dir=os.path.join(filepath, f"subject{subj_id:02d}_session{self.session_id}")
            _load_data(self.train_samples, file_dir, subj_id, self.features)
        
        if self.mode=="AE":
            for testid in test_id_list:
                for session_id in range(1, 3):
                    file_dir = os.path.join(filepath, f"subject{testid:02d}_session{session_id}")
                    _load_data(self.test_samples, file_dir, testid, self.features)
        else:
            for testid in test_id_list:
                print(f'load test dataset {testid}')
                file_dir = os.path.join(filepath, f"subject{testid:02d}_session{self.session_id}")
                _load_data(self.test_samples, file_dir, testid, self.features)
        # preprocess.
        self.train_samples = _preproess_dict(self.train_samples)
        self.test_samples = _preproess_dict(self.test_samples)
        normalizer = StandardScaler().fit(self.train_samples["x"])
        self.train_samples["x"] = normalizer.transform(self.train_samples["x"])
        self.test_samples["x"] = normalizer.transform(self.test_samples["x"])
        self.train_samples["x"]=self.train_samples["x"].reshape(-1, len(self.features)*self.feature_dim, 16, 16)
        # for d,pattern in enumerate(self.train_samples["p"]):
        #     plt.imshow(self.train_samples["x"][d][0])
        #     plt.savefig(f"/home/DATA_STOREAGE/fanjiahao/EMG/AE_project/why_it_works/images_force_normalized/{pattern}_{d}.png")
        #     plt.close()
        self.test_samples["x"] = self.test_samples["x"].reshape(-1, len(self.features)*self.feature_dim, 16, 16)
        # if self.purpose=="test":
        #     self.train_samples["x"]=np.vstack((self.train_samples["x"],self.test_samples["x"]))
        #     self.train_samples["p"]=np.hstack((self.train_samples["p"],self.test_samples["p"]))
        #     self.train_samples["s"]=np.hstack((self.train_samples["s"],self.test_samples["s"]))
        #     self.test_samples=self.train_samples
        print(f"train sample length: {len(self.train_samples['p'])} \n test sample length:{len(self.test_samples['p'])} ")
    
    def setup(self,stage=None):
        # transform
        # return train_samples, self.test_samples
        #train=Subdataset(self.train_samples)
        # total_num=len(train)
        #val_length=int(total_num*self.val_ratio)
        # train_set,val_set=random_split(train,[total_num-val_length,val_length])
        train=Subdataset(self.train_samples)
        total_num=len(train)
        val_length=int(total_num*self.val_ratio)
        # train_set,val_set=random_split(train,[total_num-val_length,val_length])
        train_set = train
        val_set = train
        self.test_dataset=Subdataset(self.test_samples)
        self.train_dataset=train_set
        self.val_dataset=val_set


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)


    def test_dataloader(self):
        if self.purpose=='test_trainset':
            return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        elif self.purpose=='test_allset':
            return DataLoader(self.train_dataset+self.test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        else: 
            return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        # return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)


class DataModule_session(LightningDataModule):
    def __init__(self, batch_size=64, data_dir=None, features=[], feature_dim=1, test_id=0, opts=None, session_id=1, mode="C", purpose="train", shuffle=True):
        super().__init__()
        self.opts=opts
        self.batch_size=batch_size
        self.data_dir=data_dir
        self.test_id=test_id
        self.val_ratio=0
        self.num_workers=getattr(opts, "num_workers", 8)
        self.session_id=session_id
        self.mode=mode
        self.purpose=purpose
        self.train_samples = {
            "x": [],
            "p": [],
            "s": []
        }
        self.test_samples = {
            "x": [],
            "p": [],
            "s": []
        }
        self.features = features
        self.feature_dim = feature_dim
        self.shuffle = shuffle
        # self.opts.labels = np.array([1, 2, 6, 7, 13, 28, 30, 31, 32, 33]) - 1 if opts.class_num==10 else np.arange(opts.class_num)

    def _split_indices_by_class(self,new_labels,num_classes,train_ratio,shuffle=False):
        index_by_class = [[] for _ in range(num_classes)]
        for idx, class_label in enumerate(new_labels):
            index_by_class[class_label].append(idx)
        train_indices = []
        test_indices = []
        # 遍历每个类别，分配训练集和测试集的索引
        for indices in index_by_class:
            if shuffle:
                print('original indices',indices)
                np.random.shuffle(indices)  # 可选：随机打乱索引，以便获得随机样本
                print('shuffled index',indices)
            split_idx = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])
        return train_indices, test_indices

    def _load_data(self, file_dir, subj_id, train_ratio=0.8, mode=None):
        selected_patterns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 
        features = scio.loadmat(f'{file_dir}/pr_feature_smooth_dynamic.mat')['feature_smooth'][0]
        labels = scio.loadmat(f'{file_dir}/label_dynamic.mat')['label'][0]
        # print(features.shape, labels.shape)

        # select patterns
        is_valid=np.isin(labels,selected_patterns)
        label=labels[is_valid]
        feature=features[is_valid]

        # select features
        feature_selected = []
        for samp in feature:
            sampfeature = []
            for feaidx in self.features:
                tempfea = samp[0,feaidx][0]
                sampfeature.append(tempfea)
            sampfeature = np.concatenate(sampfeature,axis=0)[np.newaxis,:]
            feature_selected.append(sampfeature)
        feature_selected = np.concatenate(feature_selected,axis=0)    
        ## reshape
        feature=feature_selected
        # transform label to start from 0
        label=np.array([selected_patterns.index(c) for c in label])
        train_indices, test_indices = self._split_indices_by_class(new_labels=label, num_classes=len(selected_patterns), train_ratio=train_ratio)

        if mode=='train':
            self.train_samples["x"].append(feature[train_indices,:])
            self.train_samples["p"].append(label[train_indices])
            self.train_samples["s"].append(np.ones(len(train_indices), dtype=int) * (subj_id - 1))
        elif mode=='test':
            self.test_samples["x"].append(feature[test_indices,:])
            self.test_samples["p"].append(label[test_indices])
            self.test_samples["s"].append(np.ones(len(test_indices), dtype=int) * (subj_id - 1))
        else:
            self.train_samples["x"].append(feature)
            self.train_samples["p"].append(label)
            self.train_samples["s"].append(np.ones(len(label), dtype=int) * (subj_id - 1))

        return self.train_samples,self.test_samples
    
    def prepare_data(self):
        filepath = self.data_dir
        session_id_list = [1,2]
        test_id_list = list(range(1,21))
        session_id_test = list(set(session_id_list) - set([self.session_id]))[0]

        for subj_id in test_id_list:
            # print(f'load train dataset {subj_id}')
            file_dir=os.path.join(filepath, f"subject{subj_id:02d}_session{self.session_id}")
            self._load_data(file_dir, subj_id, 0.8, 'train')
            file_dir=os.path.join(filepath, f"subject{subj_id:02d}_session{session_id_test}")
            self._load_data(file_dir, subj_id, 0.8, 'test')

        # preprocess.
        self.train_samples = _preproess_dict(self.train_samples)
        self.test_samples = _preproess_dict(self.test_samples)
        normalizer = StandardScaler().fit(self.train_samples["x"])
        self.train_samples["x"] = normalizer.transform(self.train_samples["x"])
        self.test_samples["x"] = normalizer.transform(self.test_samples["x"])
        self.train_samples["x"]=self.train_samples["x"].reshape(-1, len(self.features)*self.feature_dim, 16, 16)
        self.test_samples["x"] = self.test_samples["x"].reshape(-1, len(self.features)*self.feature_dim, 16, 16)
        # if self.purpose=="test":
        #     self.train_samples["x"]=np.vstack((self.train_samples["x"],self.test_samples["x"]))
        #     self.train_samples["p"]=np.hstack((self.train_samples["p"],self.test_samples["p"]))
        #     self.train_samples["s"]=np.hstack((self.train_samples["s"],self.test_samples["s"]))
        #     self.test_samples=self.train_samples
        print(f"train sample length: {len(self.train_samples['p'])} \n test sample length:{len(self.test_samples['p'])} ")
    
    def setup(self,stage=None):
        # transform
        # return train_samples, self.test_samples
        #train=Subdataset(self.train_samples)
        # total_num=len(train)
        #val_length=int(total_num*self.val_ratio)
        # train_set,val_set=random_split(train,[total_num-val_length,val_length])
        train=Subdataset(self.train_samples)
        total_num=len(train)
        val_length=int(total_num*self.val_ratio)
        # train_set,val_set=random_split(train,[total_num-val_length,val_length])
        train_set = train
        val_set = train
        self.test_dataset=Subdataset(self.test_samples)
        self.train_dataset=train_set
        self.val_dataset=val_set


    def train_dataloader(self):
        print('train set:', self.train_dataset.samples['x'].shape, self.train_dataset.samples['p'].shape)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)


    def test_dataloader(self):
        if self.purpose=='test_trainset':
            print('test set:', self.train_dataset.samples['x'].shape, self.train_dataset.samples['p'].shape)
            return DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        elif self.purpose=='test_allset':
            all_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
            print('test set:', len(all_dataset))
            return DataLoader(all_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        else: 
            print('test set:', self.test_dataset.samples['x'].shape, self.test_dataset.samples['p'].shape)
            return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)
        # return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=True,persistent_workers=self.num_workers > 0)

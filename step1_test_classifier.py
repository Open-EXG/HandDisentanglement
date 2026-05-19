# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,SubsetRandomSampler
from tabulate import tabulate
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from models import ANNmodel,CNNmodel,LSTMmodel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class loadDataset(Dataset):
    def __init__(self,input_data,target):
        # 归一化
        self.x_data=F.normalize(torch.from_numpy(input_data.astype(np.float32)),dim=0)  # 样本间归一化
        # reshape
        self.x_data=self.x_data.reshape(self.x_data.shape[0],2,16,16)
        self.y_data=torch.from_numpy(target)
        print(self.x_data.shape,self.y_data.shape)
        self.dataset_size = self.y_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.dataset_size

def dataset_generation(trainset=None,testset=None,batch_size=128):
    if trainset==None:
        test_loader= torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2,drop_last=False)
        return test_loader
    train_sampler = SubsetRandomSampler(list(range(trainset.__len__())))
    try:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=False, num_workers=2,drop_last=True,sampler=train_sampler)
    except:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=trainset.__len__(),
                                                shuffle=False, num_workers=2,drop_last=True,sampler=train_sampler)        
    test_loader= torch.utils.data.DataLoader(testset, batch_size=testset.__len__(),
                                              shuffle=False, num_workers=2,drop_last=False)
    return train_loader,test_loader


def train_ann(mdl,train_loader,test_loader):
    # 定义超参数
    criterion=nn.CrossEntropyLoss(label_smoothing=0.2)
    device=DEVICE
    num_epochs=400
    best_acc,running_corrects,total_samples=0,0,0

    mdl.to(device)
    # optimizer=optim.Adam(mdl.parameters(),lr=1e-3, weight_decay=1e-5)    
    optimizer=optim.Adam(mdl.parameters(),lr=3e-4, weight_decay=1e-6)    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    for epoch in range(num_epochs):
        train_loss=0
        for _, (x,y) in enumerate(train_loader):
            mdl.train()
            optimizer.zero_grad()
            x,y =x.to(device), y.to(device)
            output = mdl(x)
            loss = criterion(output, y.type(torch.int64))
            # 计算训练集上的准确率
            prediction = torch.argsort(output, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            running_corrects += top1_acc
            total_samples += x.size(0)

            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item() 
        avg_train_loss = train_loss / len(train_loader)
        try:    
            scheduler.step()
        except:
            scheduler.step(avg_train_loss)  
        epoch_acc,_,class_acc=test(mdl,test_loader,device)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # torch.save(mdl.state_dict(), f'{args.model_path}/{mdl_name}.pth')
            # print(f'current model saved to {args.model_path}/{mdl_name}.pth')
            mdl_best=mdl
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, epoch_acc is: {epoch_acc:.4f}, best_acc is: {best_acc:.4f}, class_acc is: {class_acc}")
    return mdl_best


def test(model, test_loader, device, savename=None):
    model.to(device)
    model.eval()
    running_corrects,total_samples=0,0
    all_preds,all_targets = [], []
    with torch.no_grad():
        for _, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            try:
                pred,_=model(x)
            except:
                pred = model(x)
            prediction = torch.argsort(pred, dim=-1, descending=True)
            top1_acc = torch.sum((prediction[:, 0:1] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            running_corrects += top1_acc
            all_preds.extend(np.vstack(prediction[:, 0:1].cpu()))
            all_targets.extend(y.cpu().numpy())
            total_samples += x.size(0)
        epoch_acc = running_corrects/total_samples
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    class_accuracies = cm.diagonal()/cm.sum(axis=1)
    if savename:
        np.savez(f'{savename}', pred=all_preds, label=all_targets)
        print(f'pred results saved to {savename}')
    return epoch_acc,cm,class_accuracies


def classify(train_x,train_y,test_x,test_y,type="rf",save_path='./'):
    if type=="rf":
       clf = RandomForestClassifier(n_estimators=1000, random_state=0)
       clf.fit(train_x, train_y)
       # output=clf.predict(test_x)
    elif type=="svm":
       # from sklearn import
       clf = svm.LinearSVC()
       # lin_clf = svm.LinearSVC()
       clf.fit(train_x, train_y)
       # output = lin_clf.predict(test_x)
    elif type=="knn":
       clf = KNeighborsClassifier(n_neighbors=1)
       # lin_clf = svm.LinearSVC()
       clf.fit(train_x, train_y)
       # output = lin_clf.predict(test_x)
    elif type=="ann":
       print(np.unique(train_y).shape[0])
       clf = ANNmodel(feature_dimension=512,hidden_dim=256,num_class=np.unique(train_y).shape[0])
       train_set,test_set=loadDataset(train_x, train_y),loadDataset(test_x, test_y)
       train_loader,test_loader=dataset_generation(train_set,test_set)
       clf=train_ann(clf,train_loader,test_loader)
    elif type=="cnn":
       print(np.unique(train_y).shape[0])
       clf = CNNmodel(feature_dimension=2,num_class=np.unique(train_y).shape[0])
       train_set,test_set=loadDataset(train_x, train_y),loadDataset(test_x, test_y)
       train_loader,test_loader=dataset_generation(train_set,test_set)
       clf=train_ann(clf,train_loader,test_loader)
    elif type=="lstm":
       print(np.unique(train_y).shape[0])
       clf = LSTMmodel(feature_dimension=256,num_class=np.unique(train_y).shape[0])
       train_set,test_set=loadDataset(train_x, train_y),loadDataset(test_x, test_y)
       train_loader,test_loader=dataset_generation(train_set,test_set)
       clf=train_ann(clf,train_loader,test_loader)
    
    with open(f'{save_path}/{type}.pkl', 'wb') as f:
       pickle.dump(clf, f)
    
    return clf


def cal_subacc(test_x, test_y, ys, clf):
    sub_acc = []
    for sub in list(np.unique(ys)):
        idx = [i for (i,s) in enumerate(ys) if s==sub]
        # print(f'subject {sub}: {len(idx)}')
        output = clf.predict(test_x[idx,:])
        acc_tmp = accuracy_score(test_y[idx],output)
        sub_acc.append(acc_tmp)
    output = clf.predict(test_x)
    avg_acc = accuracy_score(test_y,output)
    return sub_acc, avg_acc   


def cal_subacc_nn(test_x, test_y, ys, clf):
    device=DEVICE
    sub_acc = []
    for sub in list(np.unique(ys)):
        idx = [i for (i,s) in enumerate(ys) if s==sub]
        # print(f'subject {sub}: {len(idx)}')
        testset=loadDataset(test_x[idx,:],test_y[idx])
        test_loader=dataset_generation(testset=testset)
        acc_tmp,_,_=test(clf,test_loader,device)
        sub_acc.append(acc_tmp)
    
    testset=loadDataset(test_x,test_y)
    test_loader=dataset_generation(testset=testset)
    avg_acc,_,_=test(clf,test_loader,device)
    return sub_acc, avg_acc 


def main(method, trial_id, test_id, session_id, val_mode, root_folder, latent_dim):
    print(trial_id)
    table_header=["subject","session","acc"]
    table_content=[]
    accs=[]
    avg_accs = []
    result=np.zeros((20,2))
    method=method  #svm | neighbor | rf
    
    for i in test_id:
        accs_tmp = []
        for j in session_id:
            train_x = []
            test_x = []
            for t in trial_id:
                dim=latent_dim
                with np.load(f"{root_folder}/latent_features/trail_{t}/test_{i}/session_{j}/train_final.npz") as f:
                    xp = f["xp"]
                    xp = xp.reshape((-1,dim))
                    # pdb.set_trace()
                    xs = f["xs"]
                    xs = xs.reshape((-1,dim))
                    yp = f["yp"]
                    ys = f["ys"]
                    if val_mode=='s':
                        train_x.append(xs)
                        train_y = ys
                    else:
                        train_x.append(xp)
                        train_y = yp

                with np.load(f"{root_folder}/latent_features/trail_{t}/test_{i}/session_{j}/test_final.npz") as f:
                    xp=f["xp"]
                    xp=xp.reshape((-1,dim))
                    xs=f["xs"]
                    xs=xs.reshape((-1,dim))
                    yp = f["yp"]
                    ys = f["ys"]               
                    test_ys = ys
                    if val_mode=='s':
                        test_x.append(xs)
                        test_y = ys
                    else:
                        test_x.append(xp)
                        test_y = yp

            train_x = np.concatenate(train_x, axis=1)
            test_x = np.concatenate(test_x, axis=1)
            # pca = PCA(n_components = 512)
            # pca.fit(train_x)
            # train_x = pca.transform(train_x)
            # test_x = pca.transform(test_x)
            save_path = f'{root_folder}/models/trail_{trial_id[0]}/test_{i}/session_{j}'
            os.makedirs(save_path, exist_ok=True)
            clf = classify(train_x,train_y,test_x,test_y,type=method,save_path=save_path)

            if method in ['svm','knn','rf']:
                sub_acc,avg_acc = cal_subacc(test_x=test_x,test_y=test_y,ys=test_ys,clf=clf)
            else:
                sub_acc,avg_acc = cal_subacc_nn(test_x=test_x,test_y=test_y,ys=test_ys,clf=clf)
            accs_tmp.append(sub_acc)
            avg_accs.append(avg_acc)
            # print(accs_tmp, avg_acc)``
            result[i-1,j-1] = avg_acc
            table_content.append([str(i),str(j),"{:.4f}".format(avg_acc)])
        accs.append(accs_tmp)
    
    accs = np.hstack(accs)
    # print(accs)
    os.makedirs(f'{root_folder}/outputs', exist_ok=True)
    np.save(f'{root_folder}/outputs/{trial_id}_{method}_1_acc.npy', accs[0,:])
    # np.save(f'{ROOT_FOLDER}/outputs/{trial_id}_{method}_2_acc.npy', accs[1,:])
    mean_acc = np.mean(accs)
    std_acc = np.std(accs,axis=1)
    table_content.append(("grand","-", "{:.4f}".format(mean_acc)))

    # np.savez(os.path.join(save_dir,"result.npz"),result=result)
    print(method)
    print(tabulate(table_content, headers=table_header, tablefmt="psql"))
    print(np.mean(std_acc))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=os.environ.get('HAND_DT_ROOT', './runs'),
                        help='Root directory created by step0_main_code.py.')
    parser.add_argument('--trial_id', type=int, nargs='+', default=[1],
                        help='One or more trial ids. Multiple ids concatenate latent features.')
    parser.add_argument('--test_id', type=int, nargs='+', default=[1],
                        help='Subject ids to evaluate.')
    parser.add_argument('--session_id', type=int, nargs='+', default=[1],
                        help='Session ids to evaluate.')
    parser.add_argument('--purpose', type=str, default='p', choices=['p', 's'],
                        help='p evaluates gesture/pattern component xp; s evaluates subject component xs.')
    parser.add_argument('--methods', type=str, nargs='+', default=['knn'],
                        choices=['knn', 'svm', 'rf', 'ann', 'cnn', 'lstm'],
                        help='Classifier methods.')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Flattened latent dimension per branch. Default is 128*2*2=512.')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Torch device for neural classifiers, e.g. cuda:0 or cpu.')
    args = parser.parse_args()
    DEVICE = args.device
    for method in args.methods:
        main(method, args.trial_id, args.test_id, args.session_id, args.purpose,
             os.path.abspath(args.root_dir), args.latent_dim)

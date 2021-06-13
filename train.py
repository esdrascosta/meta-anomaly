import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import random

from datasets.mvtech import MVTecAD
from models.unet.unet_model import UNet
import learn2learn as l2l
from sklearn.metrics import roc_auc_score
from models.svdd.model import Net

from criterions.msgms import MSGMSLoss
from criterions.ssim import SSIMLoss
from utils import mean_smoothing
from torch.nn import MSELoss

import matplotlib.pyplot as plt


def auc_metric(predictions, evaluation_labels):
    # np_predictions = predictions.detach().cpu().numpy()
    np_labels = evaluation_labels.detach().cpu().numpy()
    # pdb.set_trace()
    return roc_auc_score(np_labels, predictions)

def anomaly_score(prediction, img_input):
    loss = MSGMSLoss()
    amap = loss(img_input, prediction, as_loss=False)    
    amap = mean_smoothing(amap)
    np_amap = amap.squeeze(1).detach().cpu().numpy()  
    num_data = len(np_amap)
    y_score = np_amap.reshape(num_data, -1).max(axis=1)
    return y_score

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, with_labels=False):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    
    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    # if with_labels:
    #     evaluation_labels = labels[evaluation_indices]

    adaptation_data = data[adaptation_indices]
    evaluation_data = data[evaluation_indices]

    # adaptation_data = T.RandomHorizontalFlip()(adaptation_data)
   
    # Adapt the model
    for step in range(adaptation_steps):
        predictions = learner(adaptation_data)
        adaptation_error = loss(predictions, adaptation_data)
        learner.adapt(adaptation_error, allow_nograd=True)

    # Evaluate the adapted model
    # predictions = learner(evaluation_data)
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_data)
    # pdb.set_trace()

    evaluation_auc = None
    # if with_labels:
    #     # scores = dists.detach().cpu().numpy()
    #     scores = []
    #     for i in range(predictions.size(0)):
    #         score = anomaly_score(predictions[i], evaluation_data[i])
    #         scores.append(score)
    #     evaluation_auc = auc_metric(scores, evaluation_labels) 
    return evaluation_error, evaluation_auc

def main(lr=0.005, maml_lr=0.01, iterations=500, shots=5, tps=5, fas=3, device=torch.device("cuda"), seed=123):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_set = MVTecAD(
        obj='cable',
        transform=T.Compose([
            T.Resize(128),
            T.ToTensor(),
            # T.Normalize(mean=[0.3262804, 0.4143766, 0.46666864], std=[0.1490097, 0.21071856, 0.23415886])
        ])
    )

    test_set = MVTecAD(
        obj='cable',
        mode='test',
        transform=T.Compose([
            T.Resize(128),
            T.ToTensor(),
            # T.Normalize(mean=[0.3262804, 0.4143766, 0.46666864], std=[0.1490097, 0.21071856, 0.23415886])
        ])
    )

    meta_train = l2l.data.MetaDataset(train_set)

    train_tasks = l2l.data.TaskDataset(meta_train,
                                       task_transforms=[
                                            l2l.data.transforms.KShots(meta_train, 2*shots),
                                            l2l.data.transforms.LoadData(meta_train),
                                       ],
                                       num_tasks=1000)


    meta_test = l2l.data.MetaDataset(test_set)

    test_tasks = l2l.data.TaskDataset(meta_test,
                                       task_transforms=[
                                            l2l.data.transforms.KShots(meta_test, 2*shots),
                                            l2l.data.transforms.LoadData(meta_test),
                                       ],
                                       num_tasks=1000)

    model = UNet(3,3)
    # model = models.resnet18(num_classes=512) 
    # model = Net()
    model.to(device)

    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)

    mse_loss = nn.MSELoss()
    ssim_loss = SSIMLoss()
    msgms_loss = MSGMSLoss()

    loss_func = lambda img, rec: mse_loss(img, rec) + ssim_loss(img, rec) + msgms_loss(img, rec)

    best_auc = 0
    for iteration in range(iterations):
        opt.zero_grad()
        model.train()
        meta_train_error = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            
            evaluation_error, _ = fast_adapt(train_task, learner, loss_func, fas, shots, 0, device)

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            
        meta_train_error /= tps
        
        print('[Intaration {}] Loss : {:.5f}'.format(iteration +1, meta_train_error))

        # Average the accumulated gradients and optimize
        for p in meta_model.parameters():
            p.grad.data.mul_(1.0 / tps)
        opt.step()

        # test
        # meta_test_error = 0.0
        # meta_test_auc = 0.0
        # for _ in range(tps):
        # learner = meta_model.clone()

        for _ in range(tps):
            learner = meta_model.clone()
            test_task = test_tasks.sample()
            evaluation_error, _ = fast_adapt(test_task, learner, loss_func, fas, shots, 0, device)

        scores = []
        labels = []
        model.eval()
        with torch.no_grad():
            
            for img, label in test_set:
                img = img.to(device)
                img = img.unsqueeze(0)
            
                predictions = model(img)

                score = anomaly_score(predictions, img)
                scores.append(score)
                labels.append(label)

        evaluation_auc = roc_auc_score(np.array(labels), np.array(scores))

        # meta_test_auc += evaluation_auc.item()
        
        # print('Meta Test Error', meta_test_error / tps)
        # print('Meta Test AUC', meta_test_auc / tps)

        print('Meta Test AUC ', evaluation_auc)

        # meta_test_auc_avg = meta_test_auc / tps
        meta_test_auc_avg = evaluation_auc

        if meta_test_auc_avg > best_auc:
            best_auc =  meta_test_auc_avg

    print('Best auc:')
    print(best_auc)        
if __name__ == "__main__":
    main()

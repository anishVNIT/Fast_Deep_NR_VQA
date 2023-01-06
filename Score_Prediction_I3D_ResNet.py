import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats


class VQADataset(Dataset):
    def __init__(self, features_dir='/home/user/Desktop/pytorch-i3d-master/CNN_I3D_ResNet_features_CVD2014_database/', index=None, max_len=300, feat_dim=3072, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c_i3d.npy')
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=3072, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TemporalPooling(q, tau=12, beta=0.5):
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class I3D_ResNet_VQA(nn.Module):
    def __init__(self, input_size=3072, reduced_size=128, hidden_size=32):

        super(I3D_ResNet_VQA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        input = self.ann(input)  # dimension reduction
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):  #
            qi = q[i, :int(input_length[i].numpy())]
            qi = TemporalPooling(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


epochs=100
decay_interval = int(epochs/10)
decay_ratio = 0.8

seed = 19920517
torch.manual_seed(seed)  #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

torch.utils.backcompat.broadcast_warning.enabled = True

database = 'CVD2014'
features_dir = '/home/user/Desktop/pytorch-i3d-master/CNN_I3D_ResNet_features_CVD2014_database/'  # features dir
datainfo = '/home/user/Desktop/pytorch-i3d-master/CVD2014.mat' # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
model_name = 'I3D_ResNet_VQA'
SROCC_All = []
PLCC_All = []
KROCC_All = []
RMSE_All = []

for exp_id in range(20):
    print('EXP ID: {}'.format(exp_id))
    print(database)
    print(model_name)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-EXP{}'.format(model_name, database, exp_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-EXP{}'.format(model_name, database, exp_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_ratio = 0.2
    val_ratio = 0.2
    batch_size = 16
    
    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    index = Info['index']
    index = index[:, exp_id % index.shape[1]]  
    ref_ids = Info['ref_ids'][0, :]  
    max_len = 105#int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1 - test_ratio - val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    
    scale = Info['scores'][0, :].max()  # score normalization factor
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    if test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    
    model = I3D_ResNet_VQA().to(device)  #
    
    
    
    lr = 0.0001
    weight_decay = 1e-5
    notest_during_training = True
    
    criterion = nn.L1Loss()  # L1 loss
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(reduction='mean', delta=1.35) 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay= weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)
    best_val_criterion = -1  # SROCC min
    
    for epoch in range(epochs):
        # print(epoch)
        # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)
    
        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                # scheduler.step()
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]
    
        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC
    
    # Test
    if test_ratio > 0:
        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
        SROCC_All.append(SROCC)
        PLCC_All.append(PLCC)
        KROCC_All.append(KROCC)
        RMSE_All.append(RMSE)
        
    SROCC_Mean = np.mean(SROCC_All)
    PLCC_Mean = np.mean(PLCC_All)
    KROCC_Mean = np.mean(KROCC_All)
    RMSE_Mean = np.mean(RMSE_All)
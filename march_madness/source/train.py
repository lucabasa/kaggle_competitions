__author__ = 'lucabasa'
__version__ = '1.4.0'

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from source.torch_utils import seed_everything, MMDataset, TestDataset, train_fn, valid_fn, inference_fn
import tubesml as tml

from source.torch_utils import SmoothBCEwLogits


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout, lay_4=False):
        super().__init__()
        self.dropout = dropout
        self.lay_4 = lay_4
        
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        if self.lay_4:
            self.batch_norm3 = nn.BatchNorm1d(hidden_size)
            self.dropout3 = nn.Dropout(self.dropout)
            self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm_end = nn.BatchNorm1d(hidden_size)
        self.dropout_end = nn.Dropout(self.dropout)
        self.dense_end = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x), 1e-3)
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x), 1e-3)
        
        if self.lay_4:
            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = F.leaky_relu(self.dense3(x), 1e-3)
        
        x = self.batch_norm_end(x)
        x = self.dropout_end(x)
        x = self.dense_end(x)
        
        return x
    
    
def prepare_data(train_df, valid_df, test_df, target_cols, scaling, n_quantiles,
                 g_comp, c_comp, g_feat, c_feat, pca_add, thr):
    
    train_df, valid_df, test_df = add_pca(train_df=train_df, 
                                        valid_df=valid_df, 
                                        test_df=test_df, 
                                        scaling=scaling, n_quantiles=n_quantiles,
                                        g_comp=g_comp, c_comp=c_comp, 
                                        g_feat=g_feat, c_feat=c_feat, add=pca_add)
    if pca_add:
        train_df = process_data(data=train_df, features_g=g_feat, features_c=c_feat)
        valid_df = process_data(data=valid_df, features_g=g_feat, features_c=c_feat)
        test_df = process_data(data=test_df, features_g=g_feat, features_c=c_feat)
    
    train_df, valid_df, test_df = var_tr(train_df=train_df, 
                                       valid_df=valid_df, 
                                       test_df=test_df, 
                                       thr=thr, 
                                       cat_cols=['sig_id','cp_type','cp_time','cp_dose'])
    
    train_df = train_df.drop('cp_type', axis=1)
    valid_df = valid_df.drop('cp_type', axis=1)
    test_df = test_df.drop('cp_type', axis=1)
    
    train_df['time_dose'] = train_df['cp_time'].astype(str)+train_df['cp_dose']
    valid_df['time_dose'] = valid_df['cp_time'].astype(str)+valid_df['cp_dose']
    test_df['time_dose'] = test_df['cp_time'].astype(str)+test_df['cp_dose']
    
    train_df = pd.get_dummies(train_df, columns=['cp_time','cp_dose','time_dose'])
    valid_df = pd.get_dummies(valid_df, columns=['cp_time','cp_dose','time_dose'])
    test_df = pd.get_dummies(test_df, columns=['cp_time','cp_dose','time_dose'])
    
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
    
    #scaling
    train_df, valid_df, test_df = scale_data(train=train_df, valid=valid_df, test=test_df, scaling=scaling, n_quantiles=n_quantiles)
    
    return train_df, valid_df, test_df, feature_cols

    
def run_training(train, test, target_cols, target,  
                 batch_size, hidden_size, device, early_stopping_steps, learning_rate, epochs, weight_decay, dropout, lay_4,
                 fold, seed, verbose):
    
    test_df = test.copy()
    
    seed_everything(seed)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    del train_df['kfold']
    del valid_df['kfold']
    
#     train_df, valid_df, test_df, feature_cols = prepare_data(train_df, valid_df, test_df, target_cols, scaling, n_quantiles,
#                                                              g_comp, c_comp, g_feat, c_feat, pca_add, thr)

    scl = tml.DfScaler(method='robust')
    train_df = scl.fit_transform(train_df)
    test_df = scl.transform(test_df)
    feature_cols = train_df.columns
    
    num_features=len(feature_cols)
    num_targets=len(target_cols)
    
    x_train, y_train  = train_df[feature_cols].values, target.iloc[trn_idx, :][target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, target.iloc[val_idx, :][target_cols].values
    
    train_dataset = MMDataset(x_train, y_train)
    valid_dataset = MMDataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
        dropout=dropout,
        lay_4=lay_4
    )
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=epochs, steps_per_epoch=len(trainloader))
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=0.001)
    
    early_step = 0
    oof = np.zeros((len(train), target.iloc[:, 0:].shape[1]))
    best_loss = np.inf
    train_losses = []
    valid_losses = []
    
    if verbose:
        print(f"FOLD: {fold}, n_features={num_features}")
    
    for epoch in range(epochs):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, device)
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, device)
        if verbose:
            print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"models/FOLD{fold}_.pth")
            early_step = 0
        
        elif early_stopping_steps > 0:
            early_step += 1
            if early_step >= early_stopping_steps:
                break
    
    #plot_learning(train_losses, valid_losses, fold, seed)
    
    #--------------------- PREDICTION---------------------
    x_test = test_df[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
        dropout=dropout,
        lay_4=lay_4
    )
    
    model.load_state_dict(torch.load(f"models/FOLD{fold}_.pth"))
    model.to(device)
    
    predictions = np.zeros((len(test_df), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, device)
    
    return oof, predictions
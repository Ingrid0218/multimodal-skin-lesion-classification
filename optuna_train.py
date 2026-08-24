import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 導入 Optuna
import optuna
from optuna.trial import TrialState

# 導入自訂模組
from dataset import get_df, get_transforms, MelanomaDataset
from focal_loss.focal_loss import FocalLoss
from models import Effnet_Melanoma, GradualWarmupSchedulerV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-name', type=str, default='optuna_exp')
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--enet-type', type=str, default='tf_efficientnet_b0_ns')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=15)  # 搜尋建議 10~15 epochs
    parser.add_argument('--patience', type=int, default=4)   # 【新增】Early Stopping 容忍輪數
    parser.add_argument('--n-trials', type=int, default=20)  # 搜尋次數
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--n-test', type=int, default=1) 
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def train_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    train_loss = []
    
    for (data, target) in loader:
        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta = data.to(device), meta.to(device)
            target = target.to(device).long()
            logits = model(data, meta)
        else:
            data = data.to(device)
            target = target.to(device).long()
            logits = model(data)        
        
        probs = torch.softmax(logits, dim=-1)
        loss = criterion(probs, target)
        loss.backward()

        if args.image_size in [896, 576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def val_epoch(model, loader, mel_idx, criterion, device, args):
    model.eval()
    val_loss = []
    PROBS, TARGETS = [], []
    with torch.no_grad():
        for (data, target) in loader:
            if args.use_meta:
                data, meta = data
                data, meta = data.to(device), meta.to(device)
                target = target.to(device).long()
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(args.n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data = data.to(device)
                target = target.to(device).long()
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(args.n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            
            logits /= args.n_test
            probs /= args.n_test

            PROBS.append(probs.cpu())
            TARGETS.append(target.cpu())
            loss = criterion(probs, target)
            val_loss.append(loss.item())

    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    auc_val = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    return val_loss, auc_val


def objective(trial, df_train, df_valid, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx, device, args):
    """Optuna 評估目標函數 (整合 Early Stopping 與 Pruning)"""
    
    # 1. 搜尋空間
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5)
    focal_pos_weight = trial.suggest_float('focal_pos_weight', 2.0, 8.0, step=1.0)
    warmup_multiplier = trial.suggest_int('warmup_multiplier', 5, 15)

    # 2. 模型與優化器設定
    criterion = FocalLoss(gamma=focal_gamma, weights=torch.FloatTensor([1.0, focal_pos_weight]).to(device))
    
    model = Effnet_Melanoma(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs - 1, eta_min=1e-6)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=warmup_multiplier, total_epoch=1, after_scheduler=scheduler_cosine)

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # ==================== 【Early Stopping 初始化】 ====================
    best_val_auc = 0.0
    patience_counter = 0
    # ===================================================================

    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args)
        val_loss, val_auc = val_epoch(model, valid_loader, mel_idx, criterion, device, args)

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()


        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0  # 刷新最佳成績，重置計數器
        else:
            patience_counter += 1 # 成績未提升，累計未進步輪數

        # Optuna 跨 Trial 剪枝機制 (Pruning)
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= args.patience:
            print(f"[Trial {trial.number}] Early stopping triggered at epoch {epoch}. Best AUC: {best_val_auc:.5f}")
            break

    return best_val_auc


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train, df_test, meta_features, n_meta_features, mel_idx = get_df(args.data_dir, args.use_meta)
    transforms_train, transforms_val = get_transforms(args.image_size)
    
    train_split, valid_split = train_test_split(df_train, stratify=df_train.target, test_size=0.20, random_state=42)
    df_train_sub = pd.DataFrame(train_split)
    df_valid_sub = pd.DataFrame(valid_split)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) # 前 3 輪不啟動剪枝
    )

    study.optimize(
        lambda trial: objective(
            trial, df_train_sub, df_valid_sub, meta_features, 
            n_meta_features, transforms_train, transforms_val, 
            mel_idx, device, args
        ),
        n_trials=args.n_trials
    )

    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best Trial AUC: {study.best_trial.value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print("="*50)

    study.trials_dataframe().to_csv(os.path.join(args.log_dir, f'optuna_trials_{args.save_name}.csv'), index=False)


if __name__ == '__main__':
    main()
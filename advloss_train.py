import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from dataset import get_df, get_transforms, MelanomaDataset

# 【提醒】請根據您想測試的架構更換 import 來源
# 例如: from models_AttentionBase import Effnet_Melanoma
from models import Effnet_Melanoma 

'''
Model Setting Area
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-name', type=str, default='efficientnetb2_256_20ep', help='name used for saving checkpoints and logs')
    parser.add_argument('--data-dir', type=str, default='./melanoma_external_256')
    parser.add_argument('--image-size', type=int, default=512, help='input image size (e.g. 512)')
    parser.add_argument('--enet-type', type=str, default='efficientnet-b0', help='efficientnet variant to use')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    # 【新增】損失函數控制選項
    parser.add_argument('--loss-type', type=str, default='focal', choices=['ce', 'lwce', 'focal'], 
                        help='選擇損失函數: ce (標準), lwce (加權), focal (Focal Loss)')
    parser.add_argument('--class-weights', type=str, default='1.0,5.0', 
                        help='設定類別 0 和類別 1 的懲罰權重 (支援 lwce 與 focal 的 alpha)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, 
                        help='Focal Loss 的 Gamma 參數 (預設 2.0)')
    
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

# 【新增】Focal Loss 類別實作
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # 傳入包含各類別權重的 Tensor

    def forward(self, inputs, targets):
        # 1. 計算標準 CE Loss (reduction='none' 保留每個樣本的 loss 獨立)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. 計算預測正確的機率 pt
        pt = torch.exp(-ce_loss)
        
        # 3. 計算 Focal Loss 核心公式: (1 - pt)^gamma * CE
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 4. 如果有設定 Alpha，進行類別加權 (與 LWCE 概念相同)
        if self.alpha is not None:
            alpha_t = self.alpha[targets] # 根據 target 抓取對應的權重
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)        
        
        loss = criterion(logits, target)
        loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


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


def val_epoch(model, loader, mel_idx, n_test=1):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    # 取得預測標籤與計算指標
    PREDS = PROBS.argmax(1)
    acc = (PREDS == TARGETS).mean() * 100.
    auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    
    # 計算混淆矩陣
    cm = confusion_matrix(TARGETS, PREDS)
    
    return val_loss, acc, auc, cm, PROBS, TARGETS


def run(df_train, df_valid, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):
    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = Effnet_Melanoma(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim
    )
    model = model.to(device)

    auc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.save_name}_best.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.save_name}_final.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(f"訓練集大小: {len(dataset_train)} | 驗證集大小: {len(dataset_valid)}")

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc, cm, PROBS, TARGETS = val_epoch(model, valid_loader, mel_idx)

        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}.'
        print(content)
        
        # 印出混淆矩陣
        print(f"--- Epoch {epoch} 混淆矩陣 ---")
        print(cm)
        print("-" * 30)

        with open(os.path.join(args.log_dir, f'log_{args.save_name}.txt'), 'a') as appender:
            appender.write(content + '\n')
            appender.write(f"Epoch {epoch} Confusion Matrix:\n{cm}\n\n")

        # 儲存 Epoch 1 的預測結果以供後續分析
        if epoch == 1:
            PREDS = PROBS.argmax(1)
            results_epoch1 = pd.DataFrame({
                'image_name': df_valid['image_name'].values,
                'true_label': TARGETS,
                'pred_label': PREDS,
                'prob_melanoma': PROBS[:, mel_idx]
            })
            epoch1_csv_path = os.path.join(args.log_dir, f'{args.save_name}_epoch1_predictions.csv')
            results_epoch1.to_csv(epoch1_csv_path, index=False)
            print(f"[*] 已儲存第 1 個 Epoch 的預測結果至: {epoch1_csv_path}\n")

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() 

        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc

    torch.save(model.state_dict(), model_file2)


def main():
    df_train, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.data_dir,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)
    train_split, valid_split = train_test_split(
        df_train, stratify=df_train.target, test_size=0.20, random_state=42)
    df_train = pd.DataFrame(train_split)
    df_valid = pd.DataFrame(valid_split)

    run(df_train, df_valid, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed()
    device = torch.device('cuda')
    
    # ---------------- 損失函數選擇邏輯 ---------------- #
    # 處理類別權重 (適用於 LWCE 與 Focal Loss 的 Alpha)
    weights_list = [float(w) for w in args.class_weights.split(',')]
    class_weights = torch.tensor(weights_list).float().to(device)

    if args.loss_type == 'lwce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f" 已啟用 加權交叉熵 (LWCE) | 類別權重: {weights_list}")
        
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma, reduction='mean')
        print(f" 已啟用 Focal Loss | Gamma: {args.focal_gamma} | Alpha 權重: {weights_list}")
        
    else:
        criterion = nn.CrossEntropyLoss()
        print(" 已啟用 標準交叉熵 (CE)")

    main()
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 假設這些是您的自定義模組
from dataset import get_df, get_transforms, MelanomaDataset
from efn_mlp_atten_models import Effnet_Melanoma

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--n-test', type=int, default=1) 
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

def get_trans(img, I):
    if I >= 4: img = img.transpose(2, 3)
    if I % 4 == 0: return img
    elif I % 4 == 1: return img.flip(2)
    elif I % 4 == 2: return img.flip(3)
    elif I % 4 == 3: return img.flip(2).flip(3)

# --- 新增：繪圖函數 ---
def save_analysis_plots(history, val_targets, val_probs, mel_idx,args):
    """產出合併的分析圖片"""
    plt.clf() # 清除當前圖像
    plt.figure(figsize=(15, 10))
    
    # 1. Loss & Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.legend()

    # 2. ROC Curve
    plt.subplot(2, 2, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve((val_targets == mel_idx).astype(float), val_probs[:, mel_idx])
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Class {mel_idx})')
    plt.legend()

    # 3. Precision-Recall Curve
    plt.subplot(2, 2, 4)
    precision, recall, _ = precision_recall_curve((val_targets == mel_idx).astype(float), val_probs[:, mel_idx])
    plt.plot(recall, precision, color='blue', label=f'PR Curve (AUC = {auc(recall, precision):.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (Class {mel_idx})')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(args.log_dir, f'analysis_{args.save_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"分析圖表已成功存至: {save_path}")

def train_epoch(model, loader, optimizer, criterion, device, args):
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

        train_loss.append(loss.item())
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss.item(), smooth_loss))

    return np.mean(train_loss)

def val_epoch(model, loader, mel_idx, criterion, device, args):
    model.eval()
    val_loss = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(args.n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
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
            loss = criterion(logits, target)
            val_loss.append(loss.item())

    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    PREDS = PROBS.argmax(1)

    acc = (PREDS == TARGETS).mean()
    auc_val = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    
    # 取得詳細 report 字典
    report = classification_report(TARGETS, PREDS, output_dict=True, zero_division=0)
    report_str = classification_report(TARGETS, PREDS, digits=4, zero_division=0)

    return val_loss, acc, auc_val, report, report_str, TARGETS, PREDS, PROBS

def run(df_train, df_valid, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx, device, criterion, args):

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
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    # --- Early Stopping & 檔名同步設定 ---
    patience = 10
    early_stop_counter = 0
    best_auc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    all_epochs_stats = [] # 儲存每一輪指標的清單
    
    last_targets = None
    last_probs = None

    # 統一命名的路徑
    model_file  = os.path.join(args.model_dir, f'{args.save_name}_best.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.save_name}_final.pth')
    log_file_path = os.path.join(args.log_dir, f'{args.save_name}.txt')

    for epoch in range(1, args.n_epochs + 1):
        # 獲取當前 Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        print(time.ctime(), f'Epoch {epoch} | LR: {current_lr:.7f}')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args)
        val_loss, acc, val_auc, report, report_str, targets, preds, probs = val_epoch(model, valid_loader, mel_idx, criterion, device, args)

        last_targets = targets
        last_probs = probs

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_auc'].append(val_auc)

        # --- 關鍵修正 1：補回每一輪的指標紀錄 ---
        p, r, _ = precision_recall_curve((targets == mel_idx).astype(float), probs[:, mel_idx])
        pr_auc = auc(r, p)

        epoch_stat = {
            'Epoch': epoch,
            'LR': current_lr,
            'Train_Loss': train_loss,
            'Val_Loss': val_loss,
            'Total_Acc': acc,
            'L0_Precision': report.get('0', {}).get('precision', 0),
            'L0_Recall': report.get('0', {}).get('recall', 0),
            'L0_F1': report.get('0', {}).get('f1-score', 0),
            'L1_Precision': report.get('1', {}).get('precision', 0),
            'L1_Recall': report.get('1', {}).get('recall', 0),
            'L1_F1': report.get('1', {}).get('f1-score', 0),
            'ROC_AUC': val_auc,
            'PR_AUC': pr_auc
        }
        all_epochs_stats.append(epoch_stat)

        # 寫入同步命名的文字 Log
        content = f"[{time.ctime()}] Epoch {epoch}\n"
        content += f"LR: {current_lr:.7f} | Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | Acc: {acc:.4f} | AUC: {val_auc:.6f}\n"
        content += "Classification Report:\n" + report_str + "\n" + "="*60 + "\n"
        print(content)
        
        with open(log_file_path, 'a') as f:
            f.write(content)

        # --- Early Stopping & 最佳模型/CSV 存檔邏輯 ---
        if val_auc > best_auc:
            print(f'auc_max ({best_auc:.6f} --> {val_auc:.6f}). Saving Best Model & CSV...')
            best_auc = val_auc
            early_stop_counter = 0  # 重置計數器
            
            torch.save(model.state_dict(), model_file)
            
            # 取得驗證集的 ID 欄位 (通常是第一欄)
            id_col = df_valid.columns[0]
            df_detail = pd.DataFrame({
                'ID': df_valid[id_col].values,
                'Actual': targets,
                'Prediction': preds,
                'Prob_Mel': probs[:, mel_idx]
            })
            df_detail.to_csv(os.path.join(args.log_dir, f'{args.save_name}_best_detail.csv'), index=False)
        else:
            early_stop_counter += 1
            print(f'EarlyStopping 計數器: {early_stop_counter}/{patience}')
            if early_stop_counter >= patience:
                print(f"偵測到連續 {patience} 個 Epoch 未進步，提前停止訓練。")
                break

        scheduler_warmup.step()    
        if epoch == 2: scheduler_warmup.step()

    torch.save(model.state_dict(), model_file2)
    
    # --- 關鍵修正 3：確保輸出表格 ---
    if all_epochs_stats:
        results_df = pd.DataFrame(all_epochs_stats)
        # 只對數值欄位求平均
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['Epoch'] = 'Average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        excel_path = os.path.join(args.log_dir, f'{args.save_name}_metrics.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"指標 Excel 已存至: {excel_path}")

    if last_targets is not None and last_probs is not None:
        save_analysis_plots(history, last_targets, last_probs, mel_idx, args)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    df_train, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.data_dir,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)
    train_split, valid_split = train_test_split(
        df_train, stratify=df_train.target, test_size=0.20, random_state=42)
    
    df_train_sub = pd.DataFrame(train_split)
    df_valid_sub = pd.DataFrame(valid_split)

    run(df_train_sub, df_valid_sub, meta_features, n_meta_features, 
        transforms_train, transforms_val, mel_idx, device, criterion, args)

if __name__ == '__main__':
    main()
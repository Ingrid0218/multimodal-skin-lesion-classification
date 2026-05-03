import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, precision_recall_curve, 
    average_precision_score, roc_curve
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
import torchvision.models as models
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split

# 載入你寫好的 dataset 模組
from dataset import get_df, get_transforms, MelanomaDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-name', type=str, default='cnn_baseline')
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--image-size', type=int, default=224) 
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    
    args, _ = parser.parse_known_args()
    return args

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 定義最基礎的 CNN 架構 (Vanilla CNN)
class BasicCNN_Melanoma(nn.Module):
    def __init__(self, out_dim=2):
        super(BasicCNN_Melanoma, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(64 * 4 * 4, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.classifier(x)
        return x

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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for data, target in bar:
        optimizer.zero_grad()
        
        data, target = data.to(device), target.to(device)
        logits = model(data)        
        
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return np.mean(train_loss)

def val_epoch(model, loader, mel_idx, criterion, device, epoch, args):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc=f"Validating Epoch {epoch}"):
            data, target = data.to(device), target.to(device)
            
            logits = model(data)
            probs = logits.softmax(1)

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    preds = PROBS.argmax(1) 
    
    # 這裡改成抓取「所有類別」的指標 (average=None 會回傳陣列 [Label0_val, Label1_val])
    acc = accuracy_score(TARGETS, preds)
    precisions = precision_score(TARGETS, preds, average=None, zero_division=0) 
    recalls = recall_score(TARGETS, preds, average=None, zero_division=0)
    f1s = f1_score(TARGETS, preds, average=None, zero_division=0)
    
    support_str = classification_report(TARGETS, preds, zero_division=0, digits=4)
    
    roc_aucs = []
    pr_aucs = []
    colors = ['blue', 'red']
    
    # --- 畫 ROC Curve (包含 Class 0 與 Class 1) ---
    plt.figure(figsize=(8, 6))
    for c in range(args.out_dim): 
        y_true_c = (TARGETS == c).astype(int)
        prob_c = PROBS[:, c]
        try:
            auc_c = roc_auc_score(y_true_c, prob_c)
        except ValueError:
            auc_c = 0.0
        roc_aucs.append(auc_c)
        fpr, tpr, _ = roc_curve(y_true_c, prob_c)
        plt.plot(fpr, tpr, color=colors[c], lw=2, label=f'Class {c} ROC (AUC = {auc_c:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Epoch {epoch})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, f'{args.save_name}_ROC_Curve_ep{epoch}.png'))
    plt.close()

    # --- 畫 PR Curve (包含 Class 0 與 Class 1) ---
    plt.figure(figsize=(8, 6))
    for c in range(args.out_dim): 
        y_true_c = (TARGETS == c).astype(int)
        prob_c = PROBS[:, c]
        try:
            pr_auc_c = average_precision_score(y_true_c, prob_c)
        except ValueError:
            pr_auc_c = 0.0
        pr_aucs.append(pr_auc_c)
        prec, rec, _ = precision_recall_curve(y_true_c, prob_c)
        plt.plot(rec, prec, color=colors[c], lw=2, label=f'Class {c} PR (AUC = {pr_auc_c:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Epoch {epoch})')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, f'{args.save_name}_PR_Curve_ep{epoch}.png'))
    plt.close()

    # 回傳時，把包含 Label 0 和 1 的陣列全傳出去
    return val_loss, acc, precisions, recalls, f1s, roc_aucs, pr_aucs, support_str, TARGETS

def run(df_train, df_valid, transforms_train, transforms_val, mel_idx, args, device):
    dataset_train = MelanomaDataset(df_train, 'train', meta_features=None, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features=None, transform=transforms_val)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = BasicCNN_Melanoma(out_dim=args.out_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    auc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.save_name}_best.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    training_history = []
    csv_log_path = os.path.join(args.log_dir, f'{args.save_name}_history.csv')

    for epoch in range(1, args.n_epochs + 1):
        print(f'\n{time.ctime()} Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, acc, precisions, recalls, f1s, roc_aucs, pr_aucs, support_str, _ = val_epoch(
            model, valid_loader, mel_idx, criterion, device, epoch, args
        )

        # 終端機主要顯示 Label 1 (陽性) 的進度
        content = (f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}\n'
                   f'Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}\n'
                   f'Metrics (Label 1) -> Acc: {acc:.4f}, Prec: {precisions[1]:.4f}, Rec: {recalls[1]:.4f}, F1: {f1s[1]:.4f}\n'
                   f'ROC AUC: {roc_aucs[1]:.4f} | PR AUC: {pr_aucs[1]:.4f}\n'
                   f'--- Classification Report ---\n{support_str}')
        print(content) 
        
        # 存入 CSV 時，把 0 和 1 的資訊都存下來
        training_history.append({
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]["lr"],
            'train_loss': round(float(train_loss), 4),
            'val_loss': round(float(val_loss), 4),
            'accuracy': round(float(acc), 4),
            'precision_0': round(float(precisions[0]), 4),
            'recall_0': round(float(recalls[0]), 4),
            'f1_score_0': round(float(f1s[0]), 4),
            'roc_auc_0': round(float(roc_aucs[0]), 4),
            'pr_auc_0': round(float(pr_aucs[0]), 4),
            'precision_1': round(float(precisions[1]), 4),
            'recall_1': round(float(recalls[1]), 4),
            'f1_score_1': round(float(f1s[1]), 4),
            'roc_auc_1': round(float(roc_aucs[1]), 4),
            'pr_auc_1': round(float(pr_aucs[1]), 4)
        })
        
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(csv_log_path, index=False)

        scheduler_warmup.step()    
        if epoch == 2: 
            scheduler_warmup.step()  

        # 以陽性的 ROC AUC 作為儲存最佳模型的基準
        if roc_aucs[1] > auc_max:
            print(f'Label 1 ROC AUC improved ({auc_max:.4f} --> {roc_aucs[1]:.4f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            auc_max = roc_aucs[1]

    # ==========================================
    # 訓練結束：載入最佳模型進行最終評估 (產出簡報格式表格)
    # ==========================================
    print("\n" + "="*60)
    print("Training Completed (20 Epochs)!")
    print("Loading Best Model for Final Evaluation & Generating Final PPT Table...")
    print("="*60)
    
    model.load_state_dict(torch.load(model_file))
    
    val_loss, acc, precisions, recalls, f1s, roc_aucs, pr_aucs, support_str, TARGETS = val_epoch(
        model, valid_loader, mel_idx, criterion, device, 'FINAL', args
    )
    
    # 動態計算真實的 Support 數量
    support_0 = np.sum(TARGETS == 0)
    support_1 = np.sum(TARGETS == 1)

    print(f'\n[FINAL BEST MODEL RESULTS]\n{support_str}')
    
    # 建立完美符合你 PPT 格式的 DataFrame
    best_result_df = pd.DataFrame([
        {
            'Model': 'Basic CNN',
            'Label': '0 (陰性)',
            'Accuracy': round(acc, 4),
            'Precision': round(precisions[0], 4),
            'Recall': round(recalls[0], 4),
            'F1 Score': round(f1s[0], 4),
            'Support': int(support_0),
            'ROC AUC': round(roc_aucs[0], 4),
            'PR AUC': round(pr_aucs[0], 4)
        },
        {
            'Model': 'Basic CNN',
            'Label': '1 (陽性)',
            'Accuracy': round(acc, 4),
            'Precision': round(precisions[1], 4),
            'Recall': round(recalls[1], 4),
            'F1 Score': round(f1s[1], 4),
            'Support': int(support_1),
            'ROC AUC': round(roc_aucs[1], 4),
            'PR AUC': round(pr_aucs[1], 4)
        }
    ])
    
    # 存檔
    final_csv_path = os.path.join(args.log_dir, f'{args.save_name}_final_table_format.csv')
    best_result_df.to_csv(final_csv_path, index=False)
    print(f"Final presentation table saved to: {final_csv_path}\n")
    print(best_result_df.to_string(index=False))

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train, df_test, _, _, mel_idx = get_df(args.data_dir, use_meta=False)
    transforms_train, transforms_val = get_transforms(args.image_size)
    
    train_split, valid_split = train_test_split(
        df_train, stratify=df_train.target, test_size=0.20, random_state=42)
    df_train = pd.DataFrame(train_split)
    df_valid = pd.DataFrame(valid_split)

    run(df_train, df_valid, transforms_train, transforms_val, mel_idx, args, device)

if __name__ == '__main__':
    main()
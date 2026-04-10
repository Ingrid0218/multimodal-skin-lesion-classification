import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma
from sklearn.svm import SVC
import joblib

'''
model setting area
'''
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

    parser.add_argument('--n-test', type=int, default=1) # 新增這一行
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
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader, desc="Validating"):
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

            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    PREDS = PROBS.argmax(1)

    acc = (PREDS == TARGETS).mean() * 100.
    auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    
    # 產生純文字格式的 Report
    report_str = classification_report(TARGETS, PREDS, digits=4, zero_division=0)

    # 一次性回傳所有需要的資訊
    return val_loss, acc, auc, report_str, TARGETS, PREDS

def collect_features(model, loader):
    model.eval()
    features_list = []
    targets_list = []

    with torch.no_grad():
        for (data, target) in tqdm(loader, desc="Extracting Features for SVM"):
            if args.use_meta:
                img, meta = data
                img, meta = img.to(device), meta.to(device)
                # 取得 EfficientNet 提取的影像特徵
                feat = model.extract(img).squeeze(-1).squeeze(-1) 
                # 結合 Meta 數據 (與 models.py 邏輯一致)
                meta_feat = model.meta(meta)
                combined_feat = torch.cat((feat, meta_feat), dim=1)
            else:
                img = data.to(device)
                combined_feat = model.extract(img).squeeze(-1).squeeze(-1)
            
            features_list.append(combined_feat.cpu().numpy())
            targets_list.append(target.numpy())
            
    return np.concatenate(features_list), np.concatenate(targets_list)

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
    ).to(device)

    auc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.save_name}_best.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.save_name}_final.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    print(f"Train Size: {len(dataset_train)} | Valid Size: {len(dataset_valid)}")

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')
        train_loss = train_epoch(model, train_loader, optimizer)
        
        # 執行驗證並獲取所有回傳值
        val_loss, acc, auc, report_str, targets, preds = val_epoch(model, valid_loader, mel_idx, n_test=args.n_test)

        # 整理要輸出的綜合報告字串
        content = f"\n[{time.ctime()}] Epoch {epoch}\n"
        content += f"LR: {optimizer.param_groups[0]['lr']:.7f} | Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | Acc: {acc:.4f} | AUC: {auc:.6f}\n"
        content += "Classification Report:\n"
        content += report_str + "\n"
        content += "="*60 
        
        print(content)
        
        # 將紀錄附加寫入 txt 檔案中
        log_txt_path = os.path.join(args.log_dir, f'Full_Report_{args.save_name}.txt')
        with open(log_txt_path, 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() 

        if auc > auc_max:
            print(f'auc_max ({auc_max:.6f} --> {auc:.6f}). Saving model & detail CSV...')
            torch.save(model.state_dict(), model_file)
            auc_max = auc
            
            # --- 處理精簡版 detail CSV 輸出 ---
            id_col = df_valid.columns[0]
            df_detail = pd.DataFrame({
                'ID': df_valid[id_col].values,
                'Actual': targets,
                'Prediction': preds
            })
            df_detail.to_csv(os.path.join(args.log_dir, f'detail_{args.save_name}_best.csv'), index=False)

    torch.save(model.state_dict(), model_file2)
    
    # --- SVM 決策層縫合點 ---
    print("\n[SVM Training] Starting...")
    # 1. 載入訓練過程中表現最好的 CNN 權重
    model.load_state_dict(torch.load(model_file))
    
    # 2. 提取訓練集與驗證集的特徵向量
    X_train, y_train = collect_features(model, train_loader)
    X_val, y_val = collect_features(model, valid_loader)
    
    # 3. 初始化 SVM
    svm_clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
    
    # 4. 訓練 SVM
    svm_clf.fit(X_train, y_train)
    
    # 5. 驗證 SVM 成果
    svm_acc = svm_clf.score(X_val, y_val)
    print(f"[SVM Result] Validation Accuracy: {svm_acc:.4f}")
    
    # 6. 儲存 SVM 模型
    svm_save_path = os.path.join(args.model_dir, f'{args.save_name}_svm.pkl')
    joblib.dump(svm_clf, svm_save_path)
    print(f"[SVM Saved] --> {svm_save_path}")


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
    criterion = nn.CrossEntropyLoss()

    main()

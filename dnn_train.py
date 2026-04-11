import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 設定與路徑 ---
DATA_PATH = './melanoma_external_256/train.csv'
OUTPUT_DIR = './cnn1d_analysis_results' # 更改輸出目錄以利區分
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SkinLesion1DCNN(nn.Module):
    def __init__(self, input_dim):
        super(SkinLesion1DCNN, self).__init__()
        
        # 1D-CNN 特徵萃取層
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 使用一個 Dummy Input 來自動計算經過卷積與池化後，展平(Flatten)的維度大小
        # 這樣可以避免手動計算維度造成的錯誤
        dummy_input = torch.zeros(1, 1, input_dim)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        
        # 全連接分類層
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 輸入 x 形狀原為 (Batch_Size, input_dim)
        # 擴展維度為 (Batch_Size, Channels=1, Length=input_dim) 以符合 1D-CNN 需求
        x = x.unsqueeze(1)
        
        # 經過卷積層
        x = self.features(x)
        
        # 展平 (Flatten) 成 (Batch_Size, flattened_size)
        x = x.view(x.size(0), -1) 
        
        # 經過分類層
        x = self.classifier(x)
        return x

class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def save_analysis_plots(history, val_targets, val_probs):
    """產出合併的分析圖片"""
    plt.figure(figsize=(15, 10))
    
    # 1. Loss & Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.legend()

    # 2. ROC Curve
    plt.subplot(2, 2, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(val_targets, val_probs[:, 1])
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Label 1)')
    plt.legend()

    # 3. Precision-Recall Curve
    plt.subplot(2, 2, 4)
    precision, recall, _ = precision_recall_curve(val_targets, val_probs[:, 0])
    plt.plot(recall, precision, color='blue', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve (Label 1)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_combined.png'))
    print(f"分析圖片已存至: {OUTPUT_DIR}/analysis_combined.png")

def main():
    # 資料讀取
    df = pd.read_csv(DATA_PATH)
    target = 'target'
    # 過濾掉非數值欄位 (請根據您的 CSV 調整)
    drop_cols = [target, 'image_name', 'patient_id', 'diagnosis', 'benign_malignant']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = pd.get_dummies(X).fillna(0)
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(CSVDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(CSVDataset(X_val, y_val), batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 替換為 1D-CNN 模型
    model = SkinLesion1DCNN(X_train.shape[1]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {'train_loss': [], 'val_acc': []}
    all_epochs_data = []

    for epoch in range(1, 21):
        model.train()
        losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); loss = criterion(model(inputs), labels)
            loss.backward(); optimizer.step()
            losses.append(loss.item())

        # 評估
        model.eval()
        v_probs, v_targets, v_preds = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                out = model(inputs.to(device))
                prob = torch.softmax(out, dim=1)
                v_probs.extend(prob.cpu().numpy())
                v_targets.extend(labels.numpy())
                v_preds.extend(torch.argmax(prob, dim=1).cpu().numpy())

        v_probs, v_targets, v_preds = np.array(v_probs), np.array(v_targets), np.array(v_preds)
        report = classification_report(v_targets, v_preds, output_dict=True, zero_division=0)
        
        # 提取指標 (Label 0 & 1 分開)
        p, r, _ = precision_recall_curve(v_targets, v_probs[:, 0])
        pr_auc = auc(r, p)
        roc_auc = roc_auc_score(v_targets, v_probs[:, 1])

        epoch_stat = {
            'Epoch': epoch,
            'Train_Loss': np.mean(losses),
            'Total_Acc': report['accuracy'],
            # Label 0
            'L0_Precision': report['0']['precision'], 'L0_Recall': report['0']['recall'], 
            'L0_F1': report['0']['f1-score'], 'L0_Support': report['0']['support'],
            # Label 1
            'L1_Precision': report['1']['precision'], 'L1_Recall': report['1']['recall'], 
            'L1_F1': report['1']['f1-score'], 'L1_Support': report['1']['support'],
            'ROC_AUC': roc_auc, 'PR_AUC': pr_auc
        }
        all_epochs_data.append(epoch_stat)
        history['train_loss'].append(np.mean(losses))
        history['val_acc'].append(report['accuracy'])
        print(f"Epoch {epoch} Done.")

    # 儲存 Excel
    results_df = pd.DataFrame(all_epochs_data)
    # 增加平均列
    avg_row = results_df.mean().to_dict()
    avg_row['Epoch'] = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
    results_df.to_excel(os.path.join(OUTPUT_DIR, 'detailed_metrics.xlsx'), index=False)

    # 繪圖
    save_analysis_plots(history, v_targets, v_probs)

if __name__ == '__main__':
    main()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

print("載入特徵中...")
X_train = np.load('X_train_features.npy')
y_train = np.load('y_train.npy')
X_valid = np.load('X_valid_features.npy')
y_valid = np.load('y_valid.npy')

print(f"訓練集特徵維度: {X_train.shape}")
print(f"驗證集特徵維度: {X_valid.shape}")

# 建立隨機森林分類器 (這就是你們的投票機制)
# n_estimators=500 代表由 500 棵決策樹進行投票
# class_weight='balanced' 有助於處理皮膚病變資料不平衡的問題
print("開始訓練 Random Forest...")
rf = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 預測驗證集
print("進行預測與投票...")
# predict_proba 會給出機率值，用來算 AUC
y_pred_prob = rf.predict_proba(X_valid)[:, 1]  # 假設預測目標是 index 1 (Melanoma)
y_pred = rf.predict(X_valid)

# 評估結果
auc_score = roc_auc_score(y_valid, y_pred_prob)
print(f"隨機森林 (階段三) 驗證集 AUC 表現: {auc_score:.5f}")

print("\n詳細分類報告:")
print(classification_report(y_valid, y_pred))
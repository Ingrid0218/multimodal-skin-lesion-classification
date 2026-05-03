import torch
import torch.nn as nn
import torchvision.models as models # 新增 torchvision 模型庫，用來載入 CNN

# 保留原本自定義的 Swish 函數
sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

# 全新的 CNN 模型架構
class CNN_Melanoma(nn.Module):
    def __init__(self, cnn_type='resnet50', out_dim=2, n_meta_features=0, n_meta_dim=[512, 128]):
        super(CNN_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        
        # 1. 載入預訓練的 CNN 模型 (這裡使用經典的 ResNet50)
        # 你可以透過傳入 cnn_type 決定要使用比較輕量的 resnet18 還是 resnet50
        if cnn_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
        else:
            self.cnn = models.resnet50(pretrained=True)
        
        # 2. 定義 Dropout 列表 (保留你原本設定的 5 個 Dropout)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        # 3. 取得 CNN 輸出的特徵維度 (ResNet 的最後一層分類器叫做 fc)
        in_ch = self.cnn.fc.in_features
        
        # 4. 處理 Meta 特徵 (與原版設計完全相同)
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
            
        # 5. 結合後的最後輸出層
        self.myfc = nn.Linear(in_ch, out_dim)
        
        # 6. 將 CNN 原本的分類層替換為 Identity，讓它變成純粹的特徵萃取器
        self.cnn.fc = nn.Identity()

    def extract(self, x):
        # 萃取影像特徵
        x = self.cnn(x)
        return x

    def forward(self, x, x_meta=None):
        # 直接取得萃取後的特徵，ResNet 輸出為 (Batch, Features)，所以不需要 squeeze
        x = self.extract(x)
        
        # 如果有使用中繼資料 (Meta data)，就將它與圖片特徵拼接起來
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            
        # 執行 Multi-sample Dropout 來輸出最終分類結果
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        
        return out
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        # 修正：新版 PyTorch 建議使用 saved_tensors 替代 saved_variables
        i = ctx.saved_tensors[0] 
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], cnn_out_channels=32):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = EfficientNet.from_pretrained(enet_type)
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        in_ch = self.enet._fc.in_features
        
        if n_meta_features > 0:
            # 1. 原始的 MLP 基礎特徵提取
            self.meta_cnn = nn.Sequential(
                # 第一層卷積：輸入通道為 1 (將原始 Meta 特徵當作長度為 n_meta_features 的序列)
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                Swish_Module(),
                
                # 第二層卷積：進一步提煉序列特徵並輸出指定的 cnn_out_channels
                nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_out_channels),
                Swish_Module(),
                
                # 全局平均池化：將序列長度壓縮到 1，確保不論特徵數多少，輸出形狀皆固定
                nn.AdaptiveAvgPool1d(1) 
            )
            # 最終 Meta 特徵的輸出維度就是 CNN 的輸出通道數
            in_ch += cnn_out_channels
            
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        # 提取影像特徵 -> 形狀: (Batch_Size, EfficientNet特徵數)
        x = self.extract(x).squeeze(-1).squeeze(-1)
        
        if self.n_meta_features > 0:
            # 【修改重點】Meta Data 資料流簡化
            # Step A: 直接擴展維度符合 1D-CNN 輸入要求 -> (Batch_Size, 1, n_meta_features)
            x_meta = x_meta.unsqueeze(1)
            
            # Step B: 直接通過純 1D-CNN 架構 -> (Batch_Size, cnn_out_channels, 1)
            x_meta = self.meta_cnn(x_meta)
            
            # Step C: 壓平最後一維 -> (Batch_Size, cnn_out_channels)
            x_meta = x_meta.squeeze(-1)
            
            # Step D: 將影像特徵與純 1D-CNN 提煉出的 Meta 特徵拼接
            x = torch.cat((x, x_meta), dim=1)
            
        # 執行 Multi-sample Dropout 輸出最終結果
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        
        return out

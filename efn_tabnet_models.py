import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# --- TabNet 的核心組件：GLU (Gated Linear Unit) ---
# 這是模擬決策樹「分裂與選擇」特徵的關鍵
class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        # 前一半特徵與後一半特徵透過 Sigmoid 做閘控門相乘（模擬樹的分支）
        return x[:, :x.shape[1]//2] * torch.sigmoid(x[:, x.shape[1]//2:])

class TabNetStyleMeta(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabNetStyleMeta, self).__init__()
        self.glu1 = GatedLinearUnit(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.glu2 = GatedLinearUnit(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        x = self.bn1(self.glu1(x))
        x = self.bn2(self.glu2(x))
        return x

# --- 整合進你的主模型 ---
class Effnet_TabNet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=128):
        super(Effnet_TabNet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = EfficientNet.from_pretrained(enet_type)
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        in_ch = self.enet._fc.in_features
        
        if n_meta_features > 0:
            # 將原本的 self.meta 換成 TabNet 风格的表格網路
            self.meta = TabNetStyleMeta(n_meta_features, n_meta_dim)
            in_ch += n_meta_dim
            
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet._fc = nn.Identity()

    def forward(self, x, x_meta=None):
        x = self.enet(x).squeeze(-1).squeeze(-1)
        
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta) # 透過類樹狀的神經網路提取特徵
            x = torch.cat((x, x_meta), dim=1)
            
        for i, dropout in enumerate(self.dropouts):
            if i == 0: out = self.myfc(dropout(x))
            else: out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
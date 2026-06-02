import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# --- 1. TabNet 的核心組件：GLU (Gated Linear Unit) ---
class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        # 輸出維度乘 2，一半作為主特徵，一半作為門控門 (Gate)
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        # 使用 torch.chunk 在維度 1 (特徵軸) 平分成兩塊
        res, gate = torch.chunk(x, 2, dim=1)
        return res * torch.sigmoid(gate)


# --- 2. 處理表格 Meta 資料的子網路 (TabNet 风格) ---
class TabNetStyleMetaSubNetwork(nn.Module):
    def __init__(self, input_dim, dims):
        super(TabNetStyleMetaSubNetwork, self).__init__()
        # dims 預期收到一個 list，例如 [512, 128]
        self.glu1 = GatedLinearUnit(input_dim, dims[0])
        self.bn1 = nn.BatchNorm1d(dims[0])
        self.glu2 = GatedLinearUnit(dims[0], dims[1])
        self.bn2 = nn.BatchNorm1d(dims[1])
        
    def forward(self, x):
        x = self.bn1(self.glu1(x))
        x = self.bn2(self.glu2(x))
        return x

# --- 3. 整合影像與 TabNet 表格的主模型 ---
class TabNetStyleMeta(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], **kwargs):
        super(TabNetStyleMeta, self).__init__()
        self.n_meta_features = n_meta_features
        
        # 初始化影像模型 EfficientNet
        self.enet = EfficientNet.from_pretrained(enet_type)
        
        # 【核心修正】先取出原本 _fc 的輸入特徵維度 (例如 b0 是 1280)
        in_ch = self.enet._fc.in_features
        
        # 【核心修正】取完維度後，再將分類頭替換為 nn.Identity()
        self.enet._fc = nn.Identity() 
        
        # 多重 Dropout 預防過擬合
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        
        # 如果有啟用 Meta 功能，則初始化 TabNet 表格網路
        if n_meta_features > 0:
            self.meta = TabNetStyleMetaSubNetwork(n_meta_features, n_meta_dim)
            # 最終分類器的輸入通道數 = 影像特徵數 + 表格網路最後一層輸出 (n_meta_dim[-1] = 128)
            in_ch += n_meta_dim[-1]
            
        self.myfc = nn.Linear(in_ch, out_dim)

    def forward(self, x, x_meta=None):
        # 提取影像特徵 -> (Batch_Size, EfficientNet特徵數)
        x = self.enet(x) 
        
        # 提取 TabNet 表格特徵並進行拼接
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta) 
            x = torch.cat((x, x_meta), dim=1)
            
        # 多重 Dropout 預測整合
        for i, dropout in enumerate(self.dropouts):
            if i == 0: 
                out = self.myfc(dropout(x))
            else: 
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out
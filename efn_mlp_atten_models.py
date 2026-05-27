import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# 定義 Swish 激活函數，與原版保持一致
sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):

        #i = ctx.saved_tensors[0]
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128]):
        super(Effnet_Melanoma, self).__init__()
    
        self.n_meta_features = n_meta_features
        
        # 載入預訓練的 EfficientNet 骨幹網路
        self.enet = EfficientNet.from_pretrained(enet_type)
    
        # 保留原有的 Multisample Dropout 設定
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        # 取得影像特徵維度 (例如 B0 為 1280, B2 為 1408)
        in_ch = self.enet._fc.in_features

        if n_meta_features > 0:
            # 建立 Attention/Gating 網路
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                # 修改點 1：將輸出維度精確對準影像特徵維度 in_ch
                nn.Linear(n_meta_dim[0], in_ch),
                nn.BatchNorm1d(in_ch),
                # 修改點 2：加入 Sigmoid 函數，將輸出轉換為 0 到 1 的注意力權重 (Gate)
                nn.Sigmoid() 
            )
            # 修改點 3：因為是相乘融合，分類器的輸入維度維持為 in_ch，不需要加上 n_meta_dim
        
        # 分類層輸入維度與影像特徵一致
        self.myfc = nn.Linear(in_ch, out_dim)
        
        # 移除原有的分類層
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x, x_meta=None):
        # 提取影像特徵並壓縮空間維度
        x_image = self.extract(x).squeeze(-1).squeeze(-1)

        if self.n_meta_features > 0 and x_meta is not None:
            # 修改點 4：計算 Meta 資料產生的注意力遮罩
            attn_mask = self.meta(x_meta)              
            # 修改點 5：執行 Element-wise Product (影像特徵 * 注意力遮罩)
            # 這會根據病人的 Meta 資訊，動態調整影像中不同特徵的重要性
            x = x_image * attn_mask
        else:
            x = x_image

        # 執行 Multisample Dropout 提高預測穩定度
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))         
            else:
                out += self.myfc(dropout(x))        
                
        out /= len(self.dropouts)
        return out
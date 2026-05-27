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
        self.enet = EfficientNet.from_pretrained(enet_type)
    
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        # 取得 EfficientNet 最後輸出的特徵維度 (例如 EfficientNet-B0 是 1280)
        in_ch = self.enet._fc.in_features

        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                # 【修改重點 1】：將這裡的輸出維度改為 `in_ch`，確保與影像特徵長度一致
                nn.Linear(n_meta_dim[0], in_ch),
                # 【修改重點 2】：Batch Normalization 的維度也要跟著改為 `in_ch`
                nn.BatchNorm1d(in_ch),
                Swish_Module(),
            )
            # 【修改重點 3】：移除 in_ch += n_meta_dim[1]
            # 因為我們是逐元素相乘，所以融合後的維度長度不會增加，依然是原本的 in_ch

        self.myfc = nn.Linear(in_ch, out_dim)
        
        self.enet._fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)

        if self.n_meta_features > 0:
            # 表格資料通過 MLP 萃取，現在它的維度會變得跟影像特徵 (x) 一模一樣
            x_meta = self.meta(x_meta)              
            # 【修改重點 5】：將 torch.cat 替換為乘號 (*)，執行 Element-wise Product
            x = x * x_meta                          

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))         
            else:
                out += self.myfc(dropout(x))        
                
        out /= len(self.dropouts)
        return out
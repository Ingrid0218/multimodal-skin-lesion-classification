import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# ==========================================
# 1. 基礎工具與激活函數 (自定義 Swish)
# ==========================================
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


# ==========================================
# 2. 原版模型：影像 (EfficientNet) + 表格數值 Meta
# ==========================================
class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128]):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = EfficientNet.from_pretrained(enet_type)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet._fc.in_features
        
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
            
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


# ==========================================
# 3. 多模態模型 A：影像 (EfficientNet) + 文字敘述 (DNN)
# ==========================================
class Multimodal_EffNet_TextDNN(nn.Module):
    def __init__(self, enet_type='efficientnet-b0', vocab_size=10000, embed_dim=128, text_hidden_dim=256, out_dim=2):
        super(Multimodal_EffNet_TextDNN, self).__init__()
        
        # --- 影像分支 (Image Branch) ---
        self.enet = EfficientNet.from_pretrained(enet_type)
        img_in_ch = self.enet._fc.in_features
        self.enet._fc = nn.Identity()

        # --- 文字分支 (Text Branch) ---
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.text_dnn = nn.Sequential(
            nn.Linear(embed_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(text_hidden_dim, text_hidden_dim),
            nn.ReLU()
        )

        # --- 融合與分類層 (Fusion & Classifier) ---
        fusion_dim = img_in_ch + text_hidden_dim
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, image, text=None):
        img_features = self.enet(image)
        if img_features.dim() > 2:
            img_features = img_features.squeeze(-1).squeeze(-1)

        if text is not None:
            text_emb = self.embedding(text)
            text_pooled = text_emb.mean(dim=1) 
            text_features = self.text_dnn(text_pooled)
        else:
            device = img_features.device
            out_features = self.text_dnn[3].out_features
            text_features = torch.zeros(img_features.size(0), out_features).to(device)

        combined_features = torch.cat((img_features, text_features), dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(combined_features))
            else:
                out += self.classifier(dropout(combined_features))
        out /= len(self.dropouts)
        
        return out


# ==========================================
# 4. 多模態模型 B：影像 (CNN/ResNet) + 文字敘述 (DNN)
# ==========================================
class Multimodal_CNN_TextDNN(nn.Module):
    def __init__(self, cnn_type='resnet50', vocab_size=10000, embed_dim=128, text_hidden_dim=256, out_dim=2):
        super(Multimodal_CNN_TextDNN, self).__init__()
        
        # --- 影像分支 (Image Branch - CNN) ---
        if cnn_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
        else:
            self.cnn = models.resnet50(pretrained=True)
            
        img_in_ch = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # --- 文字分支 (Text Branch - DNN) ---
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.text_dnn = nn.Sequential(
            nn.Linear(embed_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(text_hidden_dim, text_hidden_dim),
            nn.ReLU()
        )

        # --- 融合與分類層 (Fusion & Classifier) ---
        fusion_dim = img_in_ch + text_hidden_dim
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, image, text=None):
        img_features = self.cnn(image)
        if img_features.dim() > 2:
            img_features = img_features.view(img_features.size(0), -1)

        if text is not None:
            text_emb = self.embedding(text)
            text_pooled = text_emb.mean(dim=1) 
            text_features = self.text_dnn(text_pooled)
        else:
            device = img_features.device
            out_features = self.text_dnn[3].out_features
            text_features = torch.zeros(img_features.size(0), out_features).to(device)

        combined_features = torch.cat((img_features, text_features), dim=1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.classifier(dropout(combined_features))
            else:
                out += self.classifier(dropout(combined_features))
        out /= len(self.dropouts)
        
        return out
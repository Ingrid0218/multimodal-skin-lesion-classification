import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. 新增：簡單的文字處理器 (Tokenizer)
# 負責將文字轉為數字序列 (Token IDs) 並進行長度補齊
# ==========================================
class SimpleTokenizer:
    def __init__(self, vocab_size=10000, max_len=20):
        self.vocab_size = vocab_size
        self.max_len = max_len
        # 保留 0 給 Padding (補齊長度)，1 給 Unknown (沒看過的字)
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        
    def fit(self, texts):
        """根據所有文字資料建立字典"""
        all_words = []
        for text in texts:
            # 轉小寫並用空白切分
            all_words.extend(str(text).lower().split())
        
        # 取出現頻率最高的前 vocab_size - 2 個單字
        vocab = [word for word, _ in Counter(all_words).most_common(self.vocab_size - 2)]
        for idx, word in enumerate(vocab, start=2):
            self.word2idx[word] = idx
            
    def encode(self, text):
        """將單一句子轉成固定長度的數字序列"""
        words = str(text).lower().split()
        # 把單字轉換為 ID，如果不在字典裡就給 <UNK> 的 ID (1)
        seq = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # 處理長度：太長就截斷，太短就補 0 (<PAD>)
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [self.word2idx["<PAD>"]] * (self.max_len - len(seq))
            
        return seq

# ==========================================
# 2. 修改：資料集類別
# ==========================================
class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, use_text=False, text_col='text', tokenizer=None, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_text = use_text
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        # --- 處理影像 ---
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        # --- 處理文字 (如果有啟用) ---
        if self.use_text and self.tokenizer is not None:
            text_data = row[self.text_col]
            # 轉換文字為 ID 序列
            text_tokens = self.tokenizer.encode(text_data)
            # 將圖片與文字包裝成 Tuple
            data = (torch.tensor(image).float(), torch.tensor(text_tokens).long())
        else:
            data = torch.tensor(image).float()

        # --- 回傳格式 ---
        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


# ==========================================
# 3. 影像擴增 (維持原樣)
# ==========================================
def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        albumentations.AutoContrast(p=0.75),   
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),
        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.CoarseDropout(max_holes=1, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    return transforms_train, transforms_val

# ==========================================
# 4. 修改：資料讀取邏輯 (取代舊的 Meta 資料萃取)
# ==========================================
def get_df(data_dir, use_text, vocab_size=10000, max_len=20):

    # ⚠️ 這裡請填入你 CSV 裡面裝載「文字敘述」的真實欄位名稱
    # 例如：'diagnosis', 'clinical_notes', 或是你自己定義的 'text'
    TEXT_COLUMN = 'anatom_site_general_challenge' 

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'train/train', f'{x}.jpg'))

    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'test/test', f'{x}.jpg'))

    tokenizer = None
    if use_text:
        # 將空值填補為 'unknown' 以免報錯
        df_train[TEXT_COLUMN] = df_train[TEXT_COLUMN].fillna('unknown')
        df_test[TEXT_COLUMN] = df_test[TEXT_COLUMN].fillna('unknown')
        
        # 初始化 Tokenizer 並使用所有資料建立字典
        tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_len=max_len)
        all_texts = pd.concat([df_train[TEXT_COLUMN], df_test[TEXT_COLUMN]]).tolist()
        tokenizer.fit(all_texts)
        print(f"✅ 文字字典建立完成，實際字典大小: {len(tokenizer.word2idx)}")

    # class mapping
    mel_idx = 1
    
    # 回傳 5 個變數，確保跟 train.py 接軌 (第三個與第四個位置從原本的 meta 換成了 tokenizer 和 文字欄位名稱)
    return df_train, df_test, tokenizer, TEXT_COLUMN, mel_idx
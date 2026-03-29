import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    args, _ = parser.parse_known_args()
    return args

def get_features(model, loader, device, use_meta=False):
    model.eval()
    features_list = []
    targets_list = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Extracting"):
            if use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                features = model.extract_features(data, meta)
            else:
                data, target = data.to(device), target.to(device)
                features = model.extract_features(data)

            features_list.append(features.cpu().numpy())
            targets_list.append(target.cpu().numpy())

    return np.concatenate(features_list), np.concatenate(targets_list)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    device = torch.device('cuda')

    # 1. 讀取資料
    df_train, df_test, meta_features, n_meta_features, mel_idx = get_df(args.data_dir, args.use_meta)
    _, transforms_val = get_transforms(args.image_size) 
    
    train_split, valid_split = train_test_split(df_train, stratify=df_train.target, test_size=0.20, random_state=42)
    df_train = pd.DataFrame(train_split)
    df_valid = pd.DataFrame(valid_split)

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_val)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. 建立模型並載入權重
    model = Effnet_Melanoma(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim
    )
    model = model.to(device)

    model_file = os.path.join(args.model_dir, f'{args.save_name}_best.pth')
    print(f"正在載入模型權重: {model_file}")
    model.load_state_dict(torch.load(model_file))

    # 3. 提取並儲存特徵
    print("\n開始提取訓練集特徵...")
    X_train, y_train = get_features(model, train_loader, device, args.use_meta)
    
    print("\n開始提取驗證集特徵...")
    X_valid, y_valid = get_features(model, valid_loader, device, args.use_meta)

    print("\n正在儲存特徵檔案...")
    np.save('X_train_features.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_valid_features.npy', X_valid)
    np.save('y_valid.npy', y_valid)
    print("太棒了！特徵提取完畢，已成功儲存 4 個 .npy 檔案。")

if __name__ == '__main__':
    main()
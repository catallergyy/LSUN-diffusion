import os
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
from diffusers import DDPMScheduler
from diffusers.models import UNet2DModel
import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
# FID用（特徴抽出）
model_feat = InceptionV3(include_top=False, pooling='avg', weights="imagenet")
# IS用（確率予測）
model_clf = InceptionV3(include_top=True, weights="imagenet")
def fid(original,generated):
    #画像の前処理
    def to_numpy_channel_last(imgs):
        if hasattr(imgs,'cpu'):
            imgs=imgs.cpu().detach()
            if imgs.ndim==4 and imgs.shape[1] in [1,3]:
                imgs=imgs.permute(0,2,3,1)
            imgs=imgs.numpy()
        if imgs.min()<0:
            imgs=(imgs/2+0.5)*255
        return imgs
    
    original_input=to_numpy_channel_last(original)
    generated_input=to_numpy_channel_last(generated)
    
    def prepare_images(imgs):
        imgs=tf.convert_to_tensor(imgs)
        imgs=tf.image.resize(imgs,(299,299))
        imgs=preprocess_input(imgs)
        return imgs
    
    original_input=prepare_images(original_input)
    generated_input=prepare_images(generated_input)
    #特徴量抽出
    original_feature=model_feat.predict(original_input, verbose=0)
    generated_feature=model_feat.predict(generated_input, verbose=0)
    #統計量の計算
    mu_original=np.mean(original_feature, axis=0)
    mu_generated=np.mean(generated_feature,axis=0)
    sigma_original=np.cov(original_feature, rowvar=False)
    sigma_generated=np.cov(generated_feature, rowvar=False)
    epsilon=1e-6
    
    assert mu_original.shape==mu_generated.shape,"エラー処理、muの形が違えば終了"
    assert sigma_generated.shape==sigma_original.shape,"エラー処理、sigmaの形が違えば終了"
    
    sigma_original += np.eye(sigma_original.shape[0]) * epsilon
    sigma_generated += np.eye(sigma_generated.shape[0]) * epsilon
    
    diff=mu_original-mu_generated
    
    covmean=sigma_original.dot(sigma_generated)
    res=sqrtm(covmean)
    
    cov_sqrt=res[0] if isinstance(res,tuple) else res
    
    if np.iscomplexobj(cov_sqrt):
        conv_sqrt=cov_sqrt.real
    tr_covmean=np.trace(cov_sqrt)
    fid_score=np.sum(diff**2)+np.trace(sigma_original)+np.trace(sigma_generated)-2*tr_covmean
    #FIDスコア計算終了
    #IS計算
    predictions=model_clf.predict(generated_input, verbose=0)
    
    marginal_prob=np.mean(predictions, axis=0)
    epsilon_is=1e-16
    k1=predictions*(np.log(predictions+epsilon_is)-np.log(marginal_prob+epsilon_is))
    k1=np.sum(k1,axis=1)
    
    inception_score=np.exp(np.mean(k1))
    
    return fid_score,inception_score

if __name__ == "__main__":
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("実データセットから画像をロード中...")
    
    # 1. 画像の前処理（モデルに入力できる形に合わせる: [0, 1] または [-1, 1]）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1] に正規化
    ])

    # 2. データセットの読み込み（例として cat フォルダを指定、bedroom でも可）
    val_dir = "./dataset/val_256"
    dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    
    # バッチサイズ分（例: 32枚）だけ取り出すデータローダー
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 1バッチ分（実画像）を取り出す
    real_images, labels = next(iter(dataloader))
    
    print(f"ロード完了: 画像バッチの形状 = {real_images.shape}")

    try:
        # テストとして、実画像同士（または実画像 vs 少しノイズを混ぜた実画像）を比較する
        # ※同じ画像同士なら FID は 0 に近くなるはず
        print("FID & IS の計算を実行中...")
        fid_val, is_val = fid(real_images, real_images)
        
        print(f"🎉 スモークテスト成功！")
        print(f"   - FID スコア (同一画像比較): {fid_val:.4f}")
        print(f"   - IS  スコア: {is_val:.4f}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
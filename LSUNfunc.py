from diffusers import DDPMScheduler
from diffusers.models import UNet2DModel
import torch
import cv2
import numpy as np
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
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil

# FID用（特徴抽出）
model_feat = InceptionV3(include_top=False, pooling='avg', weights="imagenet")
# IS用（確率予測）
model_clf = InceptionV3(include_top=True, weights="imagenet")

def create_model(learning_rate):
    model = UNet2DModel(
        num_class_embeds=3,          # 
        sample_size=256,              #解像度
        in_channels=3,                # RGB
        out_channels=3,               # RGB
        layers_per_block=2,           # 各ブロックのResNet層数
        
        # チャンネル数を段階的に増やす
        block_out_channels=(128, 256, 256, 512),  # 表現力の豊かさ、いったんこのままでいく
        
        # ダウンサンプリング
        down_block_types=(
            "DownBlock2D",        # 96x96 -> 48x48 (最初は軽量)
            "AttnDownBlock2D",    # 48x48 -> 24x24   (Attention導入)
            "AttnDownBlock2D",    # 24x24 -> 12x12
            "AttnDownBlock2D",    # 12x12 -> 6x6     (最も抽象的)
        ),
        
        # アップサンプリング（逆順）
        up_block_types=(
            "AttnUpBlock2D",      
            "AttnUpBlock2D",      
            "AttnUpBlock2D",      
            "UpBlock2D",          
        ),
        
        # Attentionの設定
        attention_head_dim=8,         # Attentionヘッド次元
    )
    
    # 損失関数
    loss_fn = torch.nn.MSELoss()
    
    # Optimizer - AdamW（重み減衰付き）
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,  # 正則化
        betas=(0.9, 0.999)
    )
    
    # DDPMスケジューラ
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,           # ノイズスケジュールの開始
        beta_end=0.02,               # ノイズスケジュールの終了
        beta_schedule="linear",      # 線形スケジュール
        clip_sample=False,           # サンプルのクリッピングなし
    )
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model, loss_fn, optimizer, scheduler
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
        conv_sqrt=conv_sqrt.real
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
def save_images_locally(images,path):
    os.makedirs(path,exist_ok=True)
    for i, data in enumerate(images):
        # Move tensor to CPU and detach it from the computation graph
        if isinstance(data,(list,tuple)):
            img=data[0]
        else:
            img=data
        
        
        if isinstance(img, Image.Image):
            save_path = os.path.join(path, f"generated_{i}.png")
            img.save(save_path)
            continue # 次のループへ（下のTensor用処理はスキップ）

        # 3. PyTorch Tensorの場合の処理 (学習中の保存用)
        # GPUにあるかもしれないのでCPUへ
        if hasattr(img, 'cpu'):
            img = img.cpu().detach()
        
        # チャンネルが先頭(C, H, W)なら、(H, W, C)に入れ替える
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
            
        # 値が -1〜1 や 0〜1 なら 0〜255 に直す
        img_np = img.numpy()
        if img_np.min() < 0:
            img_np = (img_np / 2+0.5)
        img_np=(img_np*255).clip(0,255).astype("uint8")

        # 保存
        pil_img = Image.fromarray(img_np)
        save_path = os.path.join(path, f"image_{i}.png")
        pil_img.save(save_path)
    
    print(f"saved {len(images)} images to {path}")

def read_and_preprocess_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        # Read image
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to (299, 299)
        img_resized = cv2.resize(img_rgb, (299, 299))

        images.append(img_resized)

    return np.array(images)

def accumulate_gradients(model,grad_accumulator):
    """毎ステップ呼び出して、勾配の大きさを蓄積する"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 勾配の絶対値の平均を取得
            grad_magnitude = param.grad.abs().mean().item()
            
            if name not in grad_accumulator:
                grad_accumulator[name] = 0.0
            grad_accumulator[name] += grad_magnitude
def generate_with_cfg_batch(model, scheduler, x, labels, guidance_scale, null_label, 
                            device, show_progress=False, desc="Generating"):
    """
    Classifier-Free Guidanceを使用してバッチ生成
    """
    iterator = tqdm(scheduler.timesteps, desc=desc) if show_progress else scheduler.timesteps
    
    for t in iterator:
        with torch.no_grad():
            # Conditional prediction
            noise_pred_cond = model(x, t, class_labels=labels).sample
            
            if guidance_scale > 1.0:
                # Unconditional prediction
                null_labels = torch.full_like(labels, null_label)
                noise_pred_uncond = model(x, t, class_labels=null_labels).sample
                
                # CFG
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # デノイズステップ
            x = scheduler.step(noise_pred, t, x).prev_sample
    
    return x


def generate_samples_with_cfg(model, scheduler, device, num_samples_per_class=4, 
                              guidance_scale=3.0, null_label=2):
    """
    各クラスからサンプルを生成（CFG使用）
    """
    all_samples = []
    
    for class_id in [0, 1]:
        x = torch.randn(num_samples_per_class, 3, 96, 96).to(device)
        labels = torch.tensor([class_id] * num_samples_per_class).to(device)
        
        x = generate_with_cfg_batch(
            model, scheduler, x, labels,
            guidance_scale=guidance_scale,
            null_label=null_label,
            device=device
        )
        
        all_samples.extend(x)
    
    return all_samples


def load_data(batch_size=64, data_dir="./dataset/val_256"):
    """
    LSUNデータセット（bedroom と cat の2クラス）をロードする
    
    Args:
        batch_size: バッチサイズ
        data_dir: bedroom と cat のフォルダが含まれる親ディレクトリのパス
    
    Returns:
        train_loader: DataLoader (画像とクラスラベルを返す)
    """
    print(f"Loading LSUN data from {data_dir}...")
    
    # 学習用の変換（[-1, 1]への正規化。拡散モデルの一般的な入力形式）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 保存用の変換（[0, 1]のまま・FID計算用や可視化用）
    save_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # ImageFolder を使ってデータセットを読み込み
    # （ImageFolderはサブディレクトリ名を自動的にクラス 0, 1 に割り当てます）
    dataset = torchvision.datasets.DatasetFolder(
        root=data_dir,
        loader=torchvision.datasets.folder.default_loader,
        extensions=('jpg', 'jpeg', 'png'),
        transform=train_transform
    )
    
    print(f"  Classes found: {dataset.classes}")  # 例: ['bedroom', 'cat']
    print(f"  Total samples: {len(dataset)}")
    
    # FID計算用のオリジナル画像保存（まだ保存されていなければ）
    original_images_path = "./original_images_lsun"
    if not os.path.exists(original_images_path) or len(os.listdir(original_images_path)) == 0:
        print(f"\nSaving original images to {original_images_path}...")
        os.makedirs(original_images_path, exist_ok=True)
        
        # 一時的に保存用の変換（正規化なし）に切り替え
        dataset.transform = save_transform
        
        sample_images = []
        # 最初から数枚（例: 100枚）を保存用として抽出
        for idx in range(min(100, len(dataset))):
            img, label = dataset[idx]
            sample_images.append(img)
            
        save_images_locally(sample_images, original_images_path)
        
        # 学習用の変換に戻す
        dataset.transform = train_transform
        print("Original images saved successfully!")
    else:
        print(f"Original images already exist in {original_images_path}")
    
    # DataLoaderの作成
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader)}")
    
    return train_loader


def verify_data_normalization(train_loader):
    """
    データの正規化を確認するデバッグ関数
    """
    print("\n=== Data Normalization Check ===")
    data, labels = next(iter(train_loader))
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Data mean: {data.mean():.3f}")
    print(f"Data std: {data.std():.3f}")
    print(f"Labels: {labels[:10].tolist()}")
    print("================================\n")
    
    
    
#pytorch_fidを使用したfid関数ISは無視,FIDの計算とテスト生成を合体
def evaluate_and_calculate_fid(model, scheduler, device, save_dir, original_folder_path, guidance_scale, null_label):
    print("\n" + "="*60)
    print("Final Generation & FID Calculation (pytorch-fid)")
    print("="*60)
    
    model.eval()
    torch.cuda.empty_cache()
    
    # 生成する総枚数
    num_samples = 100
    gen_batch_size = 10
    
    # 画像を保存する一時ディレクトリ
    gen_folder_path = f"{save_dir}/fid_generated_images"
    os.makedirs(gen_folder_path, exist_ok=True)
    
    # 最終チェック用にクラスごとのサンプルを保持する辞書
    class_samples = {0: [], 1: []}
    
    img_counter = 0
    with torch.no_grad():
        with tqdm(total=num_samples, desc="Generating evaluation images") as pbar:
            for _ in range(num_samples // gen_batch_size):
                x = torch.randn(gen_batch_size, 3, 256, 256).to(device)
                # クラス0 (bedroom) と クラス1 (cat) を半々に割り当て
                labels_gen = torch.randint(0, 2, (gen_batch_size,)).to(device)
                
                # 条件付き生成
                x_gen = generate_with_cfg_batch(
                    model, scheduler, x, labels_gen,
                    guidance_scale=guidance_scale,
                    null_label=null_label,
                    device=device,
                    show_progress=False
                )
                
                # 生成画像を1枚ずつ保存＆仕分け
                for img, lbl in zip(x_gen, labels_gen):
                    # テンソルを [0, 1] の範囲にクランプして保存用にする
                    img_cpu = img.detach().cpu()
                    
                    # FID計算用のフォルダに個別保存（pytorch-fidはフォルダ内の画像を読み込むため）
                    # LSUNfunc.save_images_locallyの仕様に合わせて調整してください
                    single_img_path = os.path.join(gen_folder_path, f"img_{img_counter:04d}.png")
                    torchvision.utils.save_image(img_cpu, single_img_path, normalize=True, value_range=(-1, 1))
                    
                    # 最終チェック用（各クラス最大16枚キープ）
                    lbl_val = lbl.item()
                    if len(class_samples[lbl_val]) < 16:
                        class_samples[lbl_val].append(img_cpu)
                        
                    img_counter += 1
                
                pbar.update(gen_batch_size)
                
    # --- 1. 最終チェック用サンプルの保存（クラス別フォルダへ） ---
    final_gen_path = f"{save_dir}/raw_generate_final_check"
    for class_id, imgs in class_samples.items():
        class_name = "bedroom" if class_id == 0 else "cat"
        class_save_path = f"{final_gen_path}/class_{class_id}_{class_name}"
        os.makedirs(class_save_path, exist_ok=True)
        
        # 保存用のユーティリティを使って書き出し
        save_images_locally(imgs, class_save_path)
        print(f"✓ Saved {len(imgs)} samples for class {class_id} ({class_name})")

    # --- 2. pytorch-fid を使ったFID計算 ---
    fid_score = None
    inception_score = 0.0 # pytorch-fidはISを直接出さないため必要に応じて別算出
    
    try:
        print("\nCalculating FID via pytorch-fid...")
        # original_folder_path と生成画像フォルダを比較
        fid_score = calculate_fid_given_paths(
            [original_folder_path, gen_folder_path],
            batch_size=50,
            device=device,
            dims=2048
        )
        print(f"\n✓ FID Score (pytorch-fid): {fid_score:.2f}")
        
    except Exception as e:
        print(f"⚠ FID calculation failed: {e}")
        
    # クリーンアップ（FID用の大量の一時画像は削除）
    if os.path.exists(gen_folder_path):
        shutil.rmtree(gen_folder_path)
        print(f"✓ Cleaned up temporary FID folder.")
        
    return fid_score, inception_score
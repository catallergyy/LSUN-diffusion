import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.models import UNet2DModel
from pytorch_fid.fid_score import calculate_fid_given_paths

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


FID_REF_DIR = "./fid_ref_lsun"

def build_fid_reference(data_dir, ref_dir=FID_REF_DIR):
    """data_dir/<class>/ の画像を全クラス分 ref_dir にフラットにコピー（pytorch-fidはサブフォルダを読まないため）"""
    if os.path.exists(ref_dir) and len(os.listdir(ref_dir)) > 0:
        print(f"FID reference images already exist in {ref_dir} ({len(os.listdir(ref_dir))} images)")
        return
    os.makedirs(ref_dir, exist_ok=True)
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(class_dir, fname), os.path.join(ref_dir, f"{class_name}_{fname}"))
    print(f"Saved {len(os.listdir(ref_dir))} FID reference images to {ref_dir}")


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
    
    # FID計算用の参照画像フォルダ（bedroom と cat の両クラスを全て含める）
    # 旧版は先頭100枚(=全部bedroom)だけだったので、生成画像(半分cat)と比較できていなかった
    build_fid_reference(data_dir, FID_REF_DIR)
    
    # DataLoaderの作成
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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
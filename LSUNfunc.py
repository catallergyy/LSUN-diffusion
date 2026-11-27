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

# アーキテクチャ設定
#  v1: 旧設定（STL-10 96x96からのコピー）。128x128でattentionがあり非常に遅い。旧チェックポイント読み込み用。
#  v2: 256x256向け（DDPM論文のLSUN 256設定に近い）。6段階 256->128->64->32->16->8、attentionは16x16のみ（+mid block 8x8）。
ARCHS = {
    "v1": dict(
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ),
    "v2": dict(
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",        # 256x256
            "DownBlock2D",        # 128x128
            "DownBlock2D",        # 64x64
            "DownBlock2D",        # 32x32
            "AttnDownBlock2D",    # 16x16  (Attentionはここだけ: 256トークン)
            "DownBlock2D",        # 8x8    (mid blockにもAttentionあり)
        ),
        up_block_types=(
            "UpBlock2D",          # 8x8
            "AttnUpBlock2D",      # 16x16
            "UpBlock2D",          # 32x32
            "UpBlock2D",          # 64x64
            "UpBlock2D",          # 128x128
            "UpBlock2D",          # 256x256
        ),
        attention_head_dim=64,    # 1ヘッドあたり64次元（旧8次元は小さすぎて非効率）
    ),
}
DEFAULT_ARCH = "v2"


def watermark_layer_path(model):
    """透かし埋め込み層 = 最後のup blockの最後のResNetのconv2（v1では up_blocks.3..., v2では up_blocks.5...。どちらも128ch 3x3 = 147,456重み）"""
    return f"up_blocks.{len(model.up_blocks) - 1}.resnets.2.conv2"


def create_model(learning_rate, arch=DEFAULT_ARCH):
    model = UNet2DModel(
        num_class_embeds=3,           # 0=bedroom, 1=cat, 2=null(CFG)
        sample_size=256,              # 解像度
        in_channels=3,                # RGB
        out_channels=3,               # RGB
        layers_per_block=2,           # 各ブロックのResNet層数
        **ARCHS[arch],
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
    print(f"\nModel created (arch={arch}):")
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

@torch.no_grad()
def update_ema(ema_model, model, decay, step):
    """EMA（指数移動平均）重みの更新: ema = d*ema + (1-d)*model
    学習初期は d を小さくして早く追従させる（step が増えると decay に近づく）"""
    d = min(decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - d)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def accumulate_gradients(model,grad_accumulator):
    """毎ステップ呼び出して、勾配の大きさを蓄積する"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 勾配の絶対値の平均を取得
            grad_magnitude = param.grad.abs().mean().item()
            
            if name not in grad_accumulator:
                grad_accumulator[name] = 0.0
            grad_accumulator[name] += grad_magnitude
def make_sampling_scheduler(train_scheduler):
    """学習用DDPMスケジューラと同じノイズスケジュールで、高速なDPM-Solver++を作る（再学習不要）"""
    return DPMSolverMultistepScheduler.from_config(train_scheduler.config, algorithm_type="dpmsolver++")


def generate_with_cfg_batch(model, scheduler, x, labels, guidance_scale, null_label,
                            device, show_progress=False, desc="Generating",
                            num_inference_steps=None):
    """
    Classifier-Free Guidanceを使用してバッチ生成
    - num_inference_steps を指定すると scheduler.set_timesteps() を呼ぶ（DPM-Solverなら25〜50で十分）
      指定しなければ scheduler.timesteps をそのまま使う（DDPMなら1000ステップ）
    - cond と uncond を1回のforwardにまとめて計算（モデル呼び出し回数が半分）
    - CUDAではbf16 autocastで高速化（スケジューラの計算はfp32のまま）
    """
    if num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps, device=device)  # DPM-Solverの内部状態もリセットされる
    use_cfg = guidance_scale > 1.0
    if use_cfg:
        null_labels = torch.full_like(labels, null_label)
        labels_in = torch.cat([labels, null_labels])
    else:
        labels_in = labels

    iterator = tqdm(scheduler.timesteps, desc=desc) if show_progress else scheduler.timesteps
    use_amp = (device.type == "cuda") if isinstance(device, torch.device) else str(device).startswith("cuda")

    for t in iterator:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            x_in = torch.cat([x, x]) if use_cfg else x
            out = model(x_in, t, class_labels=labels_in).sample.float()
        if use_cfg:
            noise_pred_cond, noise_pred_uncond = out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = out
        # デノイズステップ
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x


def generate_samples_with_cfg(model, scheduler, device, num_samples_per_class=4,
                              guidance_scale=3.0, null_label=2, num_inference_steps=50):
    """
    各クラスからサンプルを生成（CFG使用, DPM-Solver++）
    """
    sample_scheduler = make_sampling_scheduler(scheduler)
    all_samples = []

    for class_id in [0, 1]:
        x = torch.randn(num_samples_per_class, 3, 256, 256).to(device)
        labels = torch.tensor([class_id] * num_samples_per_class).to(device)

        x = generate_with_cfg_batch(
            model, sample_scheduler, x, labels,
            guidance_scale=guidance_scale,
            null_label=null_label,
            device=device,
            num_inference_steps=num_inference_steps,
        )

        all_samples.extend(x)

    return all_samples


FID_REF_DIR = "./fid_ref_lsun"

def limit_gpu_memory(max_gb):
    """
    PyTorch が使える GPU メモリに上限をかける。
    DGX Spark (GB10) は CPU と GPU が同じメモリ (約122GiB) を共有しているので、
    あふれると swap が起きてマシン全体がフリーズする。上限を超えたら
    "CUDA out of memory" のエラーで止まるだけにするための安全装置。
    max_gb <= 0 なら上限なし。他のプロセス（別コンテナの学習など）の使用量は制限できない点に注意。
    """
    if max_gb is None or max_gb <= 0 or not torch.cuda.is_available():
        return
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    fraction = min(1.0, max_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"GPU memory limit: {max_gb:.0f} GiB / {total_gb:.1f} GiB ({fraction:.0%})")


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


def load_data(batch_size=64, data_dir="./dataset/val_256", hflip=True):
    """
    LSUNデータセット（bedroom と cat の2クラス）をロードする
    
    Args:
        batch_size: バッチサイズ
        data_dir: bedroom と cat のフォルダが含まれる親ディレクトリのパス
        hflip: True なら50%の確率で左右反転する（データが少ないので実質2倍に増やす）
    
    Returns:
        train_loader: DataLoader (画像とクラスラベルを返す)
    """
    print(f"Loading LSUN data from {data_dir}...")
    
    # 学習用の変換（[-1, 1]への正規化。拡散モデルの一般的な入力形式）
    # 左右反転: 部屋も猫も左右を入れ替えても自然な画像なので、データ拡張として使える
    # （FID参照画像は元ファイルのコピーなので影響しない）
    flip = [transforms.RandomHorizontalFlip(p=0.5)] if hflip else []
    train_transform = transforms.Compose(flip + [
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
    print(f"  Horizontal flip: {hflip}")
    
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
def evaluate_and_calculate_fid(model, scheduler, device, save_dir, original_folder_path, guidance_scale, null_label,
                               num_samples=2000, gen_batch_size=16, num_inference_steps=50):
    print("\n" + "="*60)
    print("Final Generation & FID Calculation (pytorch-fid)")
    print("="*60)
    
    model.eval()
    torch.cuda.empty_cache()
    
    # 高速サンプラー（DDPM 1000ステップ → DPM-Solver++ num_inference_steps ステップ）
    sample_scheduler = make_sampling_scheduler(scheduler)
    print(f"Sampler: DPM-Solver++ {num_inference_steps} steps, {num_samples} images, CFG={guidance_scale}")
    
    # 画像を保存する一時ディレクトリ
    gen_folder_path = f"{save_dir}/fid_generated_images"
    os.makedirs(gen_folder_path, exist_ok=True)
    
    # 最終チェック用にクラスごとのサンプルを保持する辞書
    class_samples = {0: [], 1: []}
    
    img_counter = 0
    with torch.no_grad():
        with tqdm(total=num_samples, desc="Generating evaluation images") as pbar:
            for _ in range((num_samples + gen_batch_size - 1) // gen_batch_size):
                x = torch.randn(gen_batch_size, 3, 256, 256).to(device)
                # クラス0 (bedroom) と クラス1 (cat) をちょうど半々に割り当て
                labels_gen = ((torch.arange(gen_batch_size) + img_counter) % 2).to(device)
                
                # 条件付き生成
                x_gen = generate_with_cfg_batch(
                    model, sample_scheduler, x, labels_gen,
                    guidance_scale=guidance_scale,
                    null_label=null_label,
                    device=device,
                    show_progress=False,
                    num_inference_steps=num_inference_steps,
                )
                
                # 生成画像を1枚ずつ保存＆仕分け
                for img, lbl in zip(x_gen, labels_gen):
                    # テンソルを [0, 1] の範囲にクランプして保存用にする
                    img_cpu = img.detach().cpu()
                    
                    # FID計算用のフォルダに個別保存（pytorch-fidはフォルダ内の画像を読み込むため）
                    # LSUNfunc.save_images_locallyの仕様に合わせて調整してください
                    single_img_path = os.path.join(gen_folder_path, f"img_{img_counter:05d}.png")
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
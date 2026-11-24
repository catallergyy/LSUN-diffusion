## LSUN (256*256) cat & bedroom 条件付き拡散モデル + 重みへの透かし埋め込み
##
## 使い方:
##   python3 LSUNtrain.py --epochs 600
##   python3 LSUNtrain.py --epochs 600 --resume content/runs/2026-11-24/checkpoint_latest.pt   # 途中から再開
##   python3 LSUNtrain.py --help                                                             # 全オプション
import argparse
import copy
import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm

import LSUNembedding
import LSUNfunc


def parse_args():
    p = argparse.ArgumentParser(description="LSUN Diffusion Model Training (cat & bedroom)")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--arch", default=LSUNfunc.DEFAULT_ARCH, choices=list(LSUNfunc.ARCHS))
    p.add_argument("--sort", action="store_true",
                   help="透かし位置: 指定なし=勾配が小さい重み (SORT=False), 指定あり=勾配が大きい重み (SORT=True)")
    # CFG
    p.add_argument("--cfg", type=float, default=3.0, help="guidance scale (評価・サンプル生成時)")
    p.add_argument("--cfg-dropout", type=float, default=0.1)
    # EMA
    p.add_argument("--ema-decay", type=float, default=0.9995,
                   help="EMA重みの減衰率。サンプル生成とFIDはEMA重みを使う")
    # 保存・確認
    p.add_argument("--save-every", type=int, default=50, help="N epochごとにチェックポイント保存")
    p.add_argument("--sample-every", type=int, default=25, help="N epochごとにサンプル画像を保存")
    p.add_argument("--sample-steps", type=int, default=50, help="DPM-Solver++ のステップ数")
    p.add_argument("--fid-samples", type=int, default=2000, help="最終FIDの生成枚数 (0でFIDなし)")
    p.add_argument("--save-dir", default=None, help="default: ./content/runs/<日付>")
    p.add_argument("--resume", default=None, help="checkpoint_latest.pt などから再開")
    return p.parse_args()


def save_loss_graph(history, title, color, path):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label=title, color=color, linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_samples(model, scheduler, device, args, null_label, path):
    """各クラス8枚を生成して1枚のグリッド画像に保存（上段: bedroom, 下段: cat）"""
    was_training = model.training
    model.eval()
    samples = LSUNfunc.generate_samples_with_cfg(
        model, scheduler, device, num_samples_per_class=8,
        guidance_scale=args.cfg, null_label=null_label, num_inference_steps=args.sample_steps,
    )
    grid = torchvision.utils.make_grid(torch.stack(samples).cpu(), nrow=8, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(grid, path)
    model.train(was_training)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    resume_state = torch.load(args.resume, map_location="cpu") if args.resume else None
    if resume_state is not None:
        save_dir = args.save_dir or os.path.dirname(args.resume)
        args.arch = resume_state["arch"]
    else:
        save_dir = args.save_dir or f"./content/runs/{now}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/samples", exist_ok=True)

    num_epochs = args.epochs
    lr = args.lr
    SORT = args.sort
    # CFG
    guidance_scale = args.cfg
    null_label = 2
    cfg_dropout_prob = args.cfg_dropout

    key = LSUNembedding.generate_secret_key(LSUNembedding.wmark_m, LSUNembedding.wmark_dim, LSUNembedding.w_key_seed, device).to(device)
    wmark = LSUNembedding.generate_watermark_pattern(LSUNembedding.wmark_dim, LSUNembedding.wmark_seed, device).to(device)

    print("LSUN Diffusion Model Training (cat & bedroom)")
    print(f"  save_dir={save_dir}  epochs={num_epochs}  arch={args.arch}  lambda_={LSUNembedding.lambda_}  SORT={SORT}")
    # データとモデルの準備
    train_loader = LSUNfunc.load_data(batch_size=args.batch_size)
    LSUNfunc.verify_data_normalization(train_loader)

    model, loss_fn, optimizer, scheduler = LSUNfunc.create_model(lr, arch=args.arch)
    model.to(device)
    loss_fn.to(device)
    # EMA重み（重みの移動平均）: 生成画像がより綺麗・安定になる。生成・FIDにはこちらを使う
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    bce_loss = torch.nn.BCELoss()
    #============ 透かし埋め込み層 =============
    layer_path = LSUNfunc.watermark_layer_path(model)  # 最後のup blockの最後のconv2（v1: up_blocks.3, v2: up_blocks.5）
    #=========================================
    target_layer = model.get_submodule(layer_path)
    print(f"layer was found: {layer_path}")
    num_weights = target_layer.weight.numel()

    # 学習状態
    start_epoch = 0
    accumulated_grad = {}
    top_indices = None
    total_loss_history, normal_loss_history, w_loss_history = [], [], []
    best_loss = float("inf")
    current_steps = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        ema_model.load_state_dict(resume_state["ema_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = resume_state["epoch"] + 1
        top_indices = resume_state["watermark_indices"]
        top_indices = top_indices.to(device) if top_indices is not None else None
        total_loss_history = resume_state["total_loss_history"]
        normal_loss_history = resume_state["normal_loss_history"]
        w_loss_history = resume_state["w_loss_history"]
        best_loss = resume_state["best_loss"]
        current_steps = resume_state["steps"]
        print(f"Resumed from {args.resume} (next epoch: {start_epoch + 1})")

    def checkpoint(epoch):
        return {
            "epoch": epoch,
            "arch": args.arch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "watermark_indices": top_indices.cpu() if top_indices is not None else None,
            "secret_key": key.cpu(),
            "target_layer_name": layer_path,
            "total_loss_history": total_loss_history,
            "normal_loss_history": normal_loss_history,
            "w_loss_history": w_loss_history,
            "best_loss": best_loss,
            "steps": current_steps,
            "guidance_scale": guidance_scale,
            "null_label": null_label,
            "args": vars(args),
        }

    trainingstart_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*60}")

        model.train()
        epoch_loss = 0
        epoch_normal_loss = 0
        epoch_wmark_loss = 0
        # epoch 0: 通常学習 + 勾配の蓄積（透かし位置の決定用）
        # epoch 1以降: 通常学習 + 透かしloss (lambda_ * BCE)
        embed_wmark = top_indices is not None

        with tqdm(total=len(train_loader), desc="Training") as pbar:
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)
                # LSUN: DatasetFolder already gives bedroom=0, cat=1 (no remap needed)

                # ★ CFG用: ランダムにnull labelを適用
                cfg_mask = torch.rand(labels.shape[0], device=device) < cfg_dropout_prob
                labels = torch.where(cfg_mask, torch.tensor(null_label, device=device), labels)

                optimizer.zero_grad()

                # Forward pass
                noise = torch.randn_like(data)
                bs = data.shape[0]
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device).long()
                noisy_images = scheduler.add_noise(data, noise, timestep)

                # モデルの予測
                output = model(noisy_images, timestep, class_labels=labels).sample
                normal_loss = loss_fn(output, noise)

                if embed_wmark:
                    #=========透かし埋め込みlossの計算======================
                    selected_weights = target_layer.weight.view(-1)[top_indices].view(1, -1)
                    extracted_wmark = torch.sigmoid(torch.matmul(selected_weights, key))
                    wmark_loss = LSUNembedding.lambda_ * bce_loss(extracted_wmark, wmark)
                    total_loss = normal_loss + wmark_loss
                    epoch_wmark_loss += wmark_loss.item()
                else:
                    total_loss = normal_loss

                # Backward pass
                total_loss.backward()

                if not embed_wmark:
                    # 勾配の絶対値を蓄積（透かし位置の決定用）
                    for name, param in target_layer.named_parameters():
                        if param.grad is not None:
                            if name not in accumulated_grad:
                                accumulated_grad[name] = torch.zeros_like(param.grad)
                            accumulated_grad[name] += param.grad.abs()

                optimizer.step()
                LSUNfunc.update_ema(ema_model, model, args.ema_decay, current_steps)

                epoch_normal_loss += normal_loss.item()
                epoch_loss += total_loss.item()
                current_steps += 1

                pbar.update(1)
                pbar.set_postfix({
                    "total_avg_loss": f"{epoch_loss/(batch_idx+1):.7f}",
                    "normal_avg_loss": f"{epoch_normal_loss/(batch_idx+1):.7f}",
                    "w_avg_loss": f"{epoch_wmark_loss/(batch_idx+1):.7f}",
                })

        # 透かし埋め込み位置の決定（最初のepochの後に1回だけ）
        if top_indices is None:
            flat_grad = accumulated_grad["weight"].flatten()
            # SORT=False: 昇順 = 勾配が小さい重み / SORT=True: 降順 = 勾配が大きい重み
            sorted_vals, sorted_indices = torch.sort(flat_grad, descending=SORT)
            if num_weights < LSUNembedding.wmark_m:
                raise ValueError(f"層のパラメータ数({num_weights})が wmark_m({LSUNembedding.wmark_m}) より少ない")
            # 上位 wmark_m 個（連続）を選択
            top_indices = sorted_indices[:LSUNembedding.wmark_m].to(device)
            accumulated_grad = {}
            print("weights was selected for watermarking")
            # モデルの重みと一緒に、透かし情報も1つのファイルにパッキングする
            torch.save({
                "model_state_dict": model.state_dict(),  # 通常のAIの重み
                "arch": args.arch,
                "watermark_indices": top_indices.cpu(),  # 透かしの場所
                "secret_key": key.cpu(),                 # 秘密鍵
                "target_layer_name": layer_path,
            }, f"{save_dir}/watermarked_model.pth")

        # エポック統計
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_normal_loss = epoch_normal_loss / len(train_loader)
        avg_w_loss = epoch_wmark_loss / len(train_loader)
        total_loss_history.append(avg_epoch_loss)
        normal_loss_history.append(avg_normal_loss)
        w_loss_history.append(avg_w_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average normal loss {avg_normal_loss:.4f}")
        print(f"  Average wmark loss {avg_w_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")

        # ベストモデルの保存（学習lossが最小のもの）
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(checkpoint(epoch), f"{save_dir}/best_model.pt")
            print(f"  ✓ Best model saved (loss: {best_loss:.4f})")

        # Loss graphの保存
        save_loss_graph(total_loss_history, "Training Total Loss", "green", f"{save_dir}/total_loss_graph.png")
        save_loss_graph(normal_loss_history, "Training Normal Loss", "red", f"{save_dir}/normal_loss_graph.png")
        save_loss_graph(w_loss_history, "Training wmark Loss", "blue", f"{save_dir}/wmark_loss_graph.png")

        # 再開用チェックポイント（毎epoch上書き）と N epochごとの保存
        torch.save(checkpoint(epoch), f"{save_dir}/checkpoint_latest.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint(epoch), f"{save_dir}/checkpoint_ep{epoch+1:04d}.pt")
            print(f"  ✓ Checkpoint saved (epoch {epoch+1})")

        # N epochごとのサンプル画像（品質の確認用）
        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == num_epochs:
            path = f"{save_dir}/samples/epoch_{epoch+1:04d}.png"
            save_samples(ema_model, scheduler, device, args, null_label, path)
            print(f"  ✓ Samples saved to {path}")

    # ========== 学習完了 ==========
    total_training_time = time.time() - trainingstart_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"Final loss: {total_loss_history[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # 最終モデルの保存
    model_path = f"{save_dir}/{num_epochs}ep_{lr}lr_cfg{guidance_scale}_wdim{LSUNembedding.wmark_dim}.pt"
    torch.save(checkpoint(num_epochs - 1), model_path)
    print(f"✓ Final model saved to {model_path}")

    # ========== 評価 & FID計算（pytorch-fid使用） ==========
    fid_score, inception_score = None, 0.0
    if args.fid_samples > 0:
        fid_score, inception_score = LSUNfunc.evaluate_and_calculate_fid(
            model=ema_model,
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            original_folder_path=LSUNfunc.FID_REF_DIR,
            guidance_scale=guidance_scale,
            null_label=null_label,
            num_samples=args.fid_samples,
            num_inference_steps=args.sample_steps,
        )

    # 記録したいファイル名
    log_file_path = f"{save_dir}/training_log.txt"
    log_content = f"""
==========================================================
[Experiment Date] {now}
==========================================================
[Hyperparameters]
Epochs       : {num_epochs}
Batch Size   : {args.batch_size}
Learning Rate: {lr}
Arch         : {args.arch}
CFG scale    : {guidance_scale}
EMA decay    : {args.ema_decay}
SORT         : {SORT}
Watermark M  : {LSUNembedding.wmark_m} (weights)
Watermark Dim: {LSUNembedding.wmark_dim}
Lambda       : {LSUNembedding.lambda_}
Seed (Key)   : {LSUNembedding.w_key_seed}
Seed (position)   : {LSUNembedding.wmark_seed}
embedding position : {layer_path}
number of parameters : {num_weights}
Training time: {hours:02d}h {minutes:02d}m {seconds:02d}s

[Results]
FID          :{fid_score}  (EMA weights, {args.fid_samples} samples, DPM-Solver++ {args.sample_steps} steps)
IS           :{inception_score}
保存場所      :{save_dir}
[Model File]
Saved Path   : {model_path}
==========================================================
"""
    # ファイルに追記モード('a')で書き込み
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_content)
    print(f"\nTraining log saved to {log_file_path}")


if __name__ == "__main__":
    main()

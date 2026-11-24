##LSUN(256*256)への変更
from tqdm import tqdm
import torch
import time
import LSUNfunc
import shutil
import matplotlib.pyplot as plt
import os
import LSUNembedding
import datetime
import wandb

def main():
    original_folder_path=LSUNfunc.FID_REF_DIR
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key=LSUNembedding.generate_secret_key(LSUNembedding.wmark_m , LSUNembedding.wmark_dim, LSUNembedding.w_key_seed,device).to(device)
    wmark=LSUNembedding.generate_watermark_pattern(LSUNembedding.wmark_dim, LSUNembedding.wmark_seed,device).to(device)
    now=datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir=f"./content/runs/{now}"
    os.makedirs(save_dir, exist_ok=True)
    batch_size=16
    num_epochs=1
    lr=1e-4
    SORT=False
    accumulated_grad={}
    #CFG
    guidance_scale=3.0
    null_label=2
    cfg_dropout_prob=0.1
    print("LSUN Diffusion Model Training (cat & bedroom)")
    # データとモデルの準備 
    train_loader = LSUNfunc.load_data(batch_size=batch_size)
    LSUNfunc.verify_data_normalization(train_loader) 
    
    model, loss_fn, optimizer, scheduler = LSUNfunc.create_model(lr)
    model.to(device)
    loss_fn.to(device)
    bce_loss=torch.nn.BCELoss()
    #============ 透かし埋め込み層 =============
    layer_path = "up_blocks.3.resnets.2.conv2" 
    #=========================================
    try:
        # その変数を使ってレイヤーの実体を取得する
        target_layer = model.get_submodule(layer_path)
        print(f"layer was found: {layer_path}")
    except AttributeError:
        print("Error: layer not found")
    num_weights=target_layer.weight.numel()
    # 学習ループ 
    total_loss_history = []
    normal_loss_history = []
    w_loss_history=[]
    current_steps = 0
    best_loss = float('inf')
    
    os.makedirs("./content", exist_ok=True)
    model_path = f"{save_dir}/{num_epochs}ep_{lr}lr_cfg{guidance_scale}_wdim{LSUNembedding.wmark_dim}.pt"
    
    trainingstart_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"{'='*60}")
        
        model.train()
        epoch_loss = 0
        epoch_normal_loss=0
        epoch_wmark_loss=0
        if epoch==0:
            with tqdm(total=len(train_loader), desc=f"Training") as pbar:
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(device)
                    labels = labels.to(device)
                
                    # LSUN: DatasetFolder already gives bedroom=0, cat=1 (no remap needed)
                    
                    # ★ CFG用: ランダムにnull labelを適用
                    cfg_mask = torch.rand(labels.shape[0], device=device) < cfg_dropout_prob
                    labels = torch.where(cfg_mask, torch.tensor(null_label, device=device), labels)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    noise = torch.randn_like(data).to(device)
                    bs = data.shape[0]
                    timestep = torch.randint(
                        0, scheduler.config.num_train_timesteps, (bs,), device=device
                    ).long()
                    noisy_images = scheduler.add_noise(data, noise, timestep)
                    
                    # モデルの予測
                    output = model(noisy_images, timestep, class_labels=labels).sample
                    loss = loss_fn(output, noise)
                    
                    # Backward pass
                    loss.backward()
                    
                    for name, param in target_layer.named_parameters():
                        if param.grad is not None:
                        # 最初のバッチの時は、勾配と同じ形の0埋めテンソルを作成
                            if name not in accumulated_grad:
                                accumulated_grad[name] = torch.zeros_like(param.grad)
                        
                        # 勾配の絶対値をどんどん足し合わせていく
                            accumulated_grad[name] += param.grad.abs()
                    optimizer.step()
                    
                    
                    epoch_loss += loss.item()
                    current_steps += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                    })
        else:
            with tqdm(total=len(train_loader), desc=f"Training") as pbar:
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(device)
                    labels = labels.to(device)
                
                    # LSUN: DatasetFolder already gives bedroom=0, cat=1 (no remap needed)
                    
                    # ★ CFG用: ランダムにnull labelを適用
                    cfg_mask = torch.rand(labels.shape[0], device=device) < cfg_dropout_prob
                    labels = torch.where(cfg_mask, torch.tensor(null_label, device=device), labels)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    noise = torch.randn_like(data).to(device)
                    bs = data.shape[0]
                    timestep = torch.randint(
                        0, scheduler.config.num_train_timesteps, (bs,), device=device
                    ).long()
                    noisy_images = scheduler.add_noise(data, noise, timestep)
                    
                    # モデルの予測
                    output = model(noisy_images, timestep, class_labels=labels).sample
                    normal_loss = loss_fn(output, noise)
                    #=========透かし埋め込みlossの計算======================
                    current_weights=target_layer.weight
                    flat_weights=current_weights.view(-1)
                    selected_weights=flat_weights[top_indices]
                    selected_weights_reshaped=selected_weights.view(1,-1)
                
                    projected=torch.matmul(selected_weights_reshaped,key)
                    extracted_wmark=torch.sigmoid(projected)
                    wmark_loss_val=bce_loss(extracted_wmark,wmark)
                
                    #損失計算
                    wmark_loss=LSUNembedding.lambda_*wmark_loss_val
                    total_loss=normal_loss+wmark_loss
                    
                    epoch_normal_loss += normal_loss.item()
                    epoch_wmark_loss +=wmark_loss.item()
                    epoch_loss += total_loss.item()
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    
                    
                    current_steps += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'total_avg_loss': f'{epoch_loss/(batch_idx+1):.7f}',
                        'normal_avg_loss': f'{epoch_normal_loss/(batch_idx+1):.7f}',
                        'w_avg_loss' : f'{epoch_wmark_loss/(batch_idx+1):.7f}'
                    })
        #透かし埋め込み位置の決定
        if epoch == 0:
            for name, grad_sum in accumulated_grad.items():
                if "weight" not in name:
                    continue
                
                flat_grad = grad_sum.flatten()
            
            # 降順ソート
                sorted_vals, sorted_indices = torch.sort(flat_grad, descending=SORT)
            
                # 容量チェックの変更：1つ飛ばしで取得するため、2倍のパラメータ数が必要
                if num_weights < LSUNembedding.wmark_m * 2:
                    print(f"Error: 層のパラメータ数({num_weights})が足りません。奇数番目の抽出には {LSUNembedding.wmark_m * 2} 個必要です。")
                    continue
                
                # スライス機能 [start:stop:step] で、0, 2, 4...番目（1位, 3位, 5位...）を抽出
                top_indices = sorted_indices[:LSUNembedding.wmark_m].to(device)
                print("weights was selected for watermarking")
                # モデルの重みと一緒に、透かし情報も1つのファイルにパッキングする
                save_data = {
                    "model_state_dict": model.state_dict(),  # 通常のAIの重み
                    "watermark_indices": top_indices.cpu(),  # 透かしの場所
                    "secret_key": key.cpu(),           # 秘密鍵
                    "target_layer_name": layer_path
                }
                torch.save(save_data, f"{save_dir}/watermarked_model.pth")
                break
 
        
        # エポック統計
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_normal_loss=epoch_normal_loss / len(train_loader)
        avg_w_loss=epoch_wmark_loss / len(train_loader)
        
        total_loss_history.append(avg_epoch_loss)
        normal_loss_history.append(avg_normal_loss)
        w_loss_history.append(avg_w_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average normal loss{avg_normal_loss:.4f}")
        print(f"  Average wmark loss{avg_w_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # ベストモデルの保存
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "./content/best_model.pt")
            print(f"  ✓ Best model saved (loss: {best_loss:.4f})")
        
        # Loss graphの保存
        plt.figure(figsize=(10, 6))
        plt.plot(total_loss_history, label="Training Loss", color="green", linewidth=2)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Total Loss", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/total_loss_graph.png", dpi=100)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(normal_loss_history, label="Training normal Loss", color="red", linewidth=2)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Normal Loss", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/normal_loss_graph.png", dpi=100)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(w_loss_history, label="Training w Loss", color="blue", linewidth=2)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training wmark Loss", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/wmark_loss_graph.png", dpi=100)
        plt.close()
        #勾配の傾向調べ、勾配を元に重みパラの選択
        
        
        
    
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
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': total_loss_history,
        'guidance_scale': guidance_scale,
        'null_label': null_label,
    }, model_path)
    print(f"✓ Final model saved to {model_path}")
    
# ========== 評価 & 高速FID計算（pytorch-fid使用） ==========
    # LSUNfunc に新しく追加した統合関数を呼び出す
    fid_score, inception_score = LSUNfunc.evaluate_and_calculate_fid(
        model=model,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        original_folder_path=original_folder_path,
        guidance_scale=guidance_scale,
        null_label=null_label
    )
# 記録したいファイル名
    log_file_path = f"{save_dir}/training_log.txt"
    
 # 記録する内容のテキスト作成
    log_content = f"""
==========================================================
[Experiment Date] {now}
==========================================================
[Hyperparameters]
Epochs       : {num_epochs}
Batch Size   : {batch_size}
Learning Rate: {lr}
Watermark M  : {LSUNembedding.wmark_m} (weights)
Watermark Dim: {LSUNembedding.wmark_dim}
Lambda       : {LSUNembedding.lambda_}
Seed (Key)   : {LSUNembedding.w_key_seed}
Seed (position)   : {LSUNembedding.wmark_seed}
embedding position : {target_layer}
number of parameters : {num_weights}

[Results]
FID          :{fid_score}
IS           :{inception_score}
保存場所      :{save_dir}
[Model File]
Saved Path   : {model_path}
==========================================================
"""

    # ファイルに追記モード('a')で書き込み
    # encoding='utf-8' にしておくと日本語を混ぜても文字化けしません
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_content)

    print(f"\nTraining log saved to {log_file_path}")
    



if __name__ == "__main__":
    main()
    

    
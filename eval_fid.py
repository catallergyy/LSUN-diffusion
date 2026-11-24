"""
学習済みチェックポイントだけでFIDを計算するスクリプト（学習なし）

例:
  python3 eval_fid.py content/runs/2026-11-17/5ep_0.0001lr_cfg3.0_wdim4096.pt
  python3 eval_fid.py <ckpt> --num-samples 200 --steps 25      # 速度確認用の短いテスト
"""
import argparse
import os
import time

import torch

import LSUNfunc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help=".pt / .pth file containing 'model_state_dict'")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50, help="DPM-Solver++ sampling steps")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--null-label", type=int, default=2)
    parser.add_argument("--data-dir", default="./dataset/val_256")
    parser.add_argument("--arch", default=None, help="v1 / v2 (default: read from checkpoint, old checkpoints = v1)")
    parser.add_argument("--no-ema", action="store_true", help="EMA重みがあっても通常の重みを使う")
    parser.add_argument("--out-dir", default=None, help="default: <ckpt dir>/eval_<timestamp>")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    arch = args.arch or state.get("arch", "v1")  # 古いチェックポイントには"arch"がない = v1
    model, _, _, scheduler = LSUNfunc.create_model(1e-4, arch=arch)
    use_ema = "ema_state_dict" in state and not args.no_ema
    model.load_state_dict(state["ema_state_dict"] if use_ema else state["model_state_dict"])
    print(f"Using {'EMA' if use_ema else 'raw'} weights")
    model.to(device)

    LSUNfunc.build_fid_reference(args.data_dir, LSUNfunc.FID_REF_DIR)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), time.strftime("eval_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    fid_score, _ = LSUNfunc.evaluate_and_calculate_fid(
        model=model,
        scheduler=scheduler,
        device=device,
        save_dir=out_dir,
        original_folder_path=LSUNfunc.FID_REF_DIR,
        guidance_scale=args.cfg,
        null_label=args.null_label,
        num_samples=args.num_samples,
        gen_batch_size=args.batch_size,
        num_inference_steps=args.steps,
    )
    elapsed = time.time() - start

    msg = (f"ckpt={args.ckpt}\nweights={'EMA' if use_ema else 'raw'}\nFID={fid_score}\nnum_samples={args.num_samples} steps={args.steps} "
           f"cfg={args.cfg}\ntime={elapsed/60:.1f} min\n")
    print("\n" + msg)
    with open(os.path.join(out_dir, "fid_result.txt"), "w", encoding="utf-8") as f:
        f.write(msg)


if __name__ == "__main__":
    main()

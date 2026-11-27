# CLAUDE.md: LSUN watermarked diffusion model

This file records how Claude understands this project and the user's instructions.
**The user should check it and correct anything wrong.** Items marked ❓ are guesses or open questions.

---

## 1. The user's instructions (how to work)

1. **Record every code change on GitHub.** Make one small commit per fix, with a message that explains *why*,
   then push to `git@github.com:catallergyy/LSUN-diffusion.git` (branch `main`).
   The user reviews each change on GitHub.
2. **Explain before changing.** Propose the plan first and wait for "yes" before editing code.
3. **Don't commit large files.** Data, images, checkpoints (`*.pt`, `*.pth`) and `*.npz` stay local (`.gitignore`).
4. **Talk to the user in Japanese** (changed 2026-11-27; before it was simple English). Code comments are in Japanese too.
5. Current priorities:
   - **Generate clearer images** (better FID). The watermark is off (`lambda_=0`) for now.
   - **Evaluation and training take too long** (FID evaluation took about 40 h). Make them faster without breaking the research.

---

## 2. What the project is

A **class-conditional diffusion model** (DDPM, 256×256) for 2 classes, **bedroom** and **cat**, with:

- **Classifier-Free Guidance (CFG)**: label dropout during training, and a guidance scale at sampling time.
- **A watermark embedded in the model weights**: a secret bit pattern is hidden in selected weights of one layer.
  The goal is to prove ownership of the model.
- **FID** (pytorch-fid) to check that image quality stays good after the watermark is added.

The research question, as Claude understands it: *can the watermark be embedded and extracted reliably while keeping FID close to the non-watermarked model?* ❓

The older STL-10 version of the same idea is in `../` (root-level scripts) and `../STL/`.

---

## 3. Files

| File | Role |
|---|---|
| `LSUNdataloader.py` | Downloads images from Hugging Face into `LSUN_origin/{bedroom,cat}`, then resizes and center-crops them to 256×256 into `dataset/val_256/{bedroom,cat}` (500 per class, 1000 total). |
| `LSUNfunc.py` | Model and scheduler creation, data loading, CFG sampling, building the FID reference folder, and FID evaluation. |
| `LSUNembedding.py` | Watermark settings and the generators for the key and the bit pattern. |
| `LSUNtrain.py` | Main script: train, then embed the watermark, save, compute FID, and write a log. |
| `eval_fid.py` | (Added by Claude.) Computes FID from a saved checkpoint without training. |
| `fid_test.py` | Older FID test (TensorFlow-based) ❓ probably no longer used. |
| `fid_stats_lsun_{bedroom,cat}.npz` | Precomputed FID statistics ❓ unknown source and Inception version. They are not used by the code. |
| `content/runs/<date>/` | Outputs: checkpoints, loss graphs, sample images, `training_log.txt`. |
| `archive/` | Old material (9.5 GB, not tracked). |

### How to run (inside the devcontainer `charming_tu`, path `/workspaces/research/diffusion-model/LSUN`)
The host Python has no `diffusers`; always run inside the container, e.g.
`docker exec -w /workspaces/research/diffusion-model/LSUN charming_tu python3 ...`
GPU: NVIDIA GB10 (PyTorch warns that capability 12.1 > 12.0, but it works).

```
python3 LSUNtrain.py --epochs 600                         # normal training (lambda_=0)
python3 LSUNtrain.py --epochs 600 --sort                  # SORT=True (largest-gradient weights)
python3 LSUNtrain.py --epochs 800 --resume content/runs/<date>/checkpoint_latest.pt   # continue a run
python3 eval_fid.py content/runs/<date>/checkpoint_ep0600.pt --steps 25 --cfg 2.0     # FID only
python3 LSUNtrain.py --help                               # all options
```

---

## 4. Pipeline in detail

### 4.1 Data
- `DatasetFolder("./dataset/val_256")` → classes sorted alphabetically: **bedroom = 0, cat = 1**.
- Normalized to [-1, 1]. Batch size 16 (`--batch-size`) → 63 batches per epoch.
- The bedroom images come from `pcuenq/lsun-bedrooms`. The **cat images are real LSUN cat images that the user collected from the web**
  (confirmed by the user). The data is fine; don't change it.
  (`download_lsun_samples_from_hf` still contains a `cats_vs_dogs` download, but that is not where the current cat data came from.)
- The dataset is small (1000 images). It is the same set used for training and for the FID reference.

### 4.2 Model (`LSUNfunc.create_model(lr, arch)`)
- diffusers `UNet2DModel`, 256×256, `num_class_embeds=3` (0 = bedroom, 1 = cat, **2 = null label for CFG**), 2 ResNet layers per block.
- **`v2` (default, new):** 6 levels 256→128→64→32→16→8, channels (128,128,256,256,512,512),
  self-attention **only at 16×16** (+ mid block 8×8), head dim 64. 114M params. Close to the DDPM paper's LSUN-256 model.
- **`v1` (old, kept to load old checkpoints):** 4 levels, attention at 128/64/32 px, head dim 8. 75M params.
  Copied from the 96×96 STL-10 model; attention at 128×128 (16,384 tokens) made it extremely slow.
- Checkpoints store `"arch"`; files without it are v1. **v1 and v2 checkpoints are not compatible.**
- Loss: MSE on the predicted noise. Optimizer: AdamW (lr 1e-4, weight decay 1e-4).
- **LR schedule:** linear warmup 1000 steps (`--warmup-steps`) → cosine decay to 0 at the last step of `--epochs`.
- **EMA weights** (`--ema-decay 0.9995`): a moving average of the weights. **Samples and FID use the EMA model.**
  The watermark loss and `watermarked_model.pth` use the raw (non-EMA) weights ❓ when lambda_>0 is used again,
  check whether detection should be done on the EMA or the raw weights.
- **bf16 autocast** for the UNet forward (loss, watermark loss and weights stay fp32). `--no-amp` turns it off.
- `DDPMScheduler` for training: 1000 steps, linear beta from 1e-4 to 0.02.
- **Sampling:** DPM-Solver++ built from the same DDPM config (`make_sampling_scheduler`), default 50 steps (`--sample-steps`).

### 4.3 CFG
- Training: each label is replaced by the null label 2 with probability `cfg_dropout_prob = 0.1`.
- Sampling: `eps = eps_uncond + s * (eps_cond - eps_uncond)`, `s = guidance_scale = 3.0`.

### 4.4 Watermark (`LSUNembedding.py` + `LSUNtrain.py`)
- Target layer: **last conv2 of the last up block** (`LSUNfunc.watermark_layer_path`):
  v1 = `up_blocks.3.resnets.2.conv2`, v2 = `up_blocks.5.resnets.2.conv2`. Both are 128→128, 3×3 → 147,456 weights.
- `wmark_m = 18000` selected weights, `wmark_dim = 4096` watermark bits.
- Secret key `K`: `randn(18000, 4096)` with seed `w_key_seed=0`.
  Watermark bits `b`: `randint(0,2,(1,4096))` with seed `wmark_seed=123`.
- **Epoch 0 (normal training only):** accumulate `|grad|` of the target layer over all batches.
  At the end of epoch 0, sort the weights by their accumulated gradient and take the first `wmark_m` as the embedding positions.
  - `SORT` (`--sort` flag) is an **experiment switch** (confirmed by the user). The research compares how the gradient size of the
    chosen weights affects watermark embedding:
    - `SORT=False` → `torch.sort(descending=False)` → **the smallest-gradient weights** (the current setting, intended)
    - `SORT=True` → **the largest-gradient weights**
    Keep this option; never hard-code one direction.
  - The code takes the first `wmark_m` weights in a row (the old comments about odd-numbered indices / "×2 capacity" were never implemented and are removed).
  - Saves `watermarked_model.pth` = {model weights *at the end of epoch 0*, indices, secret key, layer name}.
    The bit pattern itself is not saved; it is regenerated from the seed.
- **Epoch ≥ 1:** loss = `MSE + lambda_ * BCE(sigmoid(w_selected · K), b)`.
- Extraction / detection: `sigmoid(w_selected · K) > 0.5` should equal `b` ❓ The detection scripts (`../detection*.py`) have not been read yet.
- `lambda_ = 0` **on purpose** (confirmed by the user): the **current goal is clearer generated images**,
  so the watermark is off for now. Don't change `lambda_` unless asked.
- The number of epochs must be **flexible**: `--epochs N` (default 600). Earlier runs used **600 epochs**.

### 4.5 Saving and evaluation
- Output folder: `content/runs/<date>/` (or `--save-dir`). Each epoch: loss graphs, `checkpoint_latest.pt` (for `--resume`),
  `best_model.pt` (lowest training loss). Every `--save-every 50` epochs: `checkpoint_epNNNN.pt`.
  Every `--sample-every 25` epochs: `samples/epoch_NNNN.png` (top row bedroom, bottom row cat, EMA weights) → **look at these to see quality improve**.
- ⚠️ Disk: one checkpoint ≈ 1.8 GB (model + EMA + optimizer). 600 epochs with `--save-every 50` ≈ 25 GB.
- Final FID (`evaluate_and_calculate_fid`): generates `--fid-samples 2000` images (exactly 50% bedroom / 50% cat) with the EMA model,
  runs pytorch-fid against `./fid_ref_lsun` (all 1000 real images), then deletes the generated images.
  Also saves 16 samples per class to `raw_generate_final_check/`. IS is not computed (always 0.0).
- Logs hyperparameters, training time and FID to `<save_dir>/training_log.txt`.
- FID with 1000 reference images is noisy: good for comparing runs, but not comparable to numbers in papers.

### 4.6 Measured speed (GB10, batch 16, 2026-11-24)
| | train step | 1 sampling step (16 images with CFG) |
|---|---|---|
| v2 + bf16 | **1.9 s** → ~130 s/epoch → **600 epochs ≈ 22 h** | 1.5 s |
| v2 fp32 | 2.5 s | 1.9 s |
| v1 (old) + bf16 | > 60 s (benchmark stopped) | — |

- Final FID with 2000 images × 50 steps ≈ 2.6 h. With `--steps 25` (or `--sample-steps 25`) ≈ 1.3 h.
- The old evaluation (v1, 1000 DDPM steps, 2 forward passes per step) explains the ~40 h.

## 5. Changes made by Claude (see the GitHub commits)

| Commit | Change |
|---|---|
| `ac23bbd` | **Bug fix:** removed the STL-10 label remap (`labels==9 → 1`). It turned every cat into label 0 (bedroom), so the cat class was never learned. **All checkpoints before this fix have this bug.** |
| `1382d64` | **Bug fix:** the FID reference was the first 100 dataset images, all bedrooms. It is now all 1000 images of both classes (`./fid_ref_lsun`). |
| `c06d6f8` | Removed TensorFlow, the two unused InceptionV3 models and the unused code; DataLoader `num_workers=4`. |
| `93e1e54` | Sampling: DPM-Solver++ with 50 steps (instead of 1000 DDPM steps), cond and uncond in one forward pass, bf16. Number of FID samples 100 → 2000 (100 is too few for a valid FID). |
| `ca510d5` | `eval_fid.py` for evaluating a checkpoint. |
| `57bc28d` | New UNet architecture **v2** (attention only at 16×16). `v1` is kept for old checkpoints. |
| `b37958e` | `LSUNtrain.py` refactor: command-line options (`--epochs` …), checkpoints, sample images, `--resume`; the two training loops are merged into one. |
| `a0e6a4a` | **EMA** weights; samples and FID use them. |
| `2e905d7` | **LR warmup + cosine decay**. |
| `c6c8023` | **bf16 autocast** + TF32 for training. |

✅ Tested on 2026-11-24 in the devcontainer: a 2-epoch run went through training → watermark position selection →
checkpoints → sample grids → FID → log without errors.

### First long run: `content/runs/2026-11-24/` (v2, 600 epochs, lambda_=0, CFG 3.0)
- Finished. **FID = 144.2** (EMA, 2000 samples, DPM-Solver++ 50 steps). Training time **35 h 54 m** (the estimate was ~22 h; cause not checked yet ❓).
- `samples/epoch_0600.png`: bedrooms are recognizable but have strong color casts (all pink / all blue) and noisy texture.
  Cats mostly fail (color blobs, only a few fur/face shapes).
  The over-saturated colors suggest the CFG scale 3.0 is too high ❓.

---

## 6. Next steps (proposed, not done)

1. ~~User starts a real run~~ → done (see above).
2. **(Running, started 2026-11-27)** On `2026-11-24/checkpoint_ep0600.pt`, find the best CFG scale by FID (e.g. 1.5 / 2 / 3 / 5) with `eval_fid.py --cfg`.
3. Choose the "best model" by FID or the last EMA checkpoint, not by training loss (the diffusion loss barely tells image quality).
4. Possible further speed-ups (not tried): `torch.compile`, channels_last.

## 7. Questions for the user ❓
- ~~Is choosing the *smallest*-gradient weights (`SORT=False`) intended?~~ → **Yes.** `SORT` is an experiment option for comparing small vs. large gradients (answered 2026-11-24).
- ~~Is `lambda_=0` / 1 epoch a baseline run, or a mistake?~~ → **On purpose.** 
(answered 2026-11-24).
- ~~Is it OK that `cats_vs_dogs` is used as the "cat" class?~~ → **The cat data is real LSUN cat images collected by the user. It's OK** (answered 2026-11-24).
- Where do the `fid_stats_lsun_*.npz` files come from?

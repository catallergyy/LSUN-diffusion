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
4. Write in simple English. Code comments in this project are in Japanese, so keep that style.
5. Current priority: **evaluation and training take too long** (FID evaluation took about 40 h). Make it faster without breaking the research.

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

---

## 4. Pipeline in detail

### 4.1 Data
- `DatasetFolder("./dataset/val_256")` → classes sorted alphabetically: **bedroom = 0, cat = 1**.
- Normalized to [-1, 1]. Batch size 16.
- ⚠️ **The "cat" images do not come from LSUN.** `LSUNdataloader.py` takes the first 500 images of the HF dataset `cats_vs_dogs`
  **without filtering by label**, so the cat folder may contain **dogs** ❓ This should be checked.
  The bedroom images come from `pcuenq/lsun-bedrooms`.
- The dataset is small (1000 images). It is the same set used for training and for the FID reference.

### 4.2 Model (`LSUNfunc.create_model`)
- diffusers `UNet2DModel`, 256×256, `num_class_embeds=3` (0 = bedroom, 1 = cat, **2 = null label for CFG**).
- Channels (128, 256, 256, 512), 2 ResNet layers per block.
- Down blocks: `DownBlock2D` (256 px) → `AttnDownBlock2D` (128 px) → `AttnDownBlock2D` (64 px) → `AttnDownBlock2D` (32 px).
  - ⚠️ **Self-attention at 128×128 = 16,384 tokens.** This is the main reason training and sampling are slow.
    The config was copied from the 96×96 STL-10 version (the comments still say 96→48).
  - `attention_head_dim=8` → many tiny heads (inefficient).
- Loss: MSE on the predicted noise. Optimizer: AdamW (lr 1e-4, weight decay 1e-4).
- `DDPMScheduler`: 1000 steps, linear beta from 1e-4 to 0.02.

### 4.3 CFG
- Training: each label is replaced by the null label 2 with probability `cfg_dropout_prob = 0.1`.
- Sampling: `eps = eps_uncond + s * (eps_cond - eps_uncond)`, `s = guidance_scale = 3.0`.

### 4.4 Watermark (`LSUNembedding.py` + `LSUNtrain.py`)
- Target layer: `up_blocks.3.resnets.2.conv2` (last conv of the last up block, 128→128, 3×3 → 147,456 weights).
- `wmark_m = 18000` selected weights, `wmark_dim = 4096` watermark bits.
- Secret key `K`: `randn(18000, 4096)` with seed `w_key_seed=0`.
  Watermark bits `b`: `randint(0,2,(1,4096))` with seed `wmark_seed=123`.
- **Epoch 0 (normal training only):** accumulate `|grad|` of the target layer over all batches.
  At the end of epoch 0, sort the weights by their accumulated gradient and take the first `wmark_m` as the embedding positions.
  - `SORT=False` → `torch.sort(descending=False)` → **the weights with the smallest gradients** are chosen
    (weights that matter least for the main task, so the watermark should hurt quality less) ❓ Claude assumes this is intentional.
  - ⚠️ Some comments don't match the code: one says "降順ソート" (descending), and others say odd-numbered indices / "×2 capacity",
    but the code takes the first `m` in a row.
  - Saves `watermarked_model.pth` = {model weights *at the end of epoch 0*, indices, secret key, layer name}.
    The bit pattern itself is not saved; it is regenerated from the seed.
- **Epoch ≥ 1:** loss = `MSE + lambda_ * BCE(sigmoid(w_selected · K), b)`.
- Extraction / detection: `sigmoid(w_selected · K) > 0.5` should equal `b` ❓ The detection scripts (`../detection*.py`) have not been read yet.
- ⚠️ Current settings disable the watermark: `lambda_ = 0`, and `num_epochs = 1` means no epoch ≥ 1 ever runs.
  Claude assumes this is a **baseline run (no watermark)** ❓

### 4.5 Evaluation (`evaluate_and_calculate_fid`)
- Generates images (50% bedroom, 50% cat), saves them as PNG, runs pytorch-fid against `./fid_ref_lsun` (all 1000 real images), then deletes the generated images.
  Also saves 16 samples per class to `raw_generate_final_check/`.
- IS is not computed (always 0.0).
- Logs the hyperparameters and FID to `content/runs/<date>/training_log.txt`.

---

## 5. Changes made by Claude (see the GitHub commits)

| Commit | Change |
|---|---|
| `ac23bbd` | **Bug fix:** removed the STL-10 label remap (`labels==9 → 1`). It turned every cat into label 0 (bedroom), so the cat class was never learned. **All checkpoints before this fix have this bug.** |
| `1382d64` | **Bug fix:** the FID reference was the first 100 dataset images, all bedrooms. It is now all 1000 images of both classes (`./fid_ref_lsun`). |
| `c06d6f8` | Removed TensorFlow, the two unused InceptionV3 models and the unused code; DataLoader `num_workers=4`. |
| `93e1e54` | Sampling: DPM-Solver++ with 50 steps (instead of 1000 DDPM steps), cond and uncond in one forward pass, bf16. Number of FID samples 100 → 2000 (100 is too few for a valid FID). |
| `ca510d5` | `eval_fid.py` for evaluating a checkpoint. |

⚠️ **These changes have not been run yet**: the host Python has no `diffusers` and the GPU was busy. The user should test first:
`python3 eval_fid.py <ckpt> --num-samples 200 --steps 25`

---

## 6. Next steps (proposed, not done)

1. User runs the short test above and reports the time and any errors.
2. **Step 2:** change the UNet to remove attention at 128 px (keep it at ≤32 px only) and use bigger heads → retrain. Old checkpoints will not load into the new architecture.
3. Merge the duplicated training loop (epoch 0 vs. epoch ≥ 1).
4. Check that the cat data contains only cats.
5. Note: FID with 1000 reference images is still noisy. It is good for comparing runs, but not comparable to numbers in papers.

## 7. Questions for the user ❓
- Is choosing the *smallest*-gradient weights (`SORT=False`) intended?
- Is `lambda_=0` / 1 epoch a baseline run, or a mistake?
- Is it OK that `cats_vs_dogs` (possibly with dogs) is used as the "cat" class instead of LSUN cat?
- Where do the `fid_stats_lsun_*.npz` files come from?

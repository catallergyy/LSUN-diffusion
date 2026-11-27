# 仕組みの詳しい説明

LSUN 透かし入り拡散モデルの処理の流れ。ルールと注意点は [`../CLAUDE.md`](../CLAUDE.md)、実験結果は [`experiments.md`](experiments.md) を参照。

---

## 1. データ

- `DatasetFolder("./dataset/val_256")` → クラスはアルファベット順: **bedroom = 0, cat = 1**。
- [-1, 1] に正規化。バッチサイズ 16（`--batch-size`）→ 1エポック 63 バッチ。
- **左右反転**（`RandomHorizontalFlip(p=0.5)`）でデータを実質 2 倍にしている（デフォルトでオン、`--no-hflip` でオフ。2026-11-27 から）。
- bedroom 画像は `pcuenq/lsun-bedrooms` から。**cat 画像はユーザーが Web から集めた本物の LSUN cat 画像**（ユーザー確認済み）。
  （`download_lsun_samples_from_hf` には `cats_vs_dogs` のダウンロードが残っているが、今の cat データはそこから来たものではない。）
- データセットは小さい（1000枚、約 0.2 GB）。学習と FID の参照に同じ画像を使っている。

## 2. モデル（`LSUNfunc.create_model(lr, arch)`）

- diffusers の `UNet2DModel`、256×256、`num_class_embeds=3`（0 = bedroom, 1 = cat, **2 = CFG 用の空ラベル**）、各ブロックに ResNet 層 2つ。
- **`v2`（デフォルト）:** 6段階 256→128→64→32→16→8、チャネル (128,128,256,256,512,512)、
  self-attention は **16×16 だけ**（+ 中間ブロック 8×8）、head dim 64。1億1400万パラメータ。DDPM 論文の LSUN-256 モデルに近い。
- **`v1`（旧、古いチェックポイントを読むためだけに残している）:** 4段階、attention は 128/64/32 px、head dim 8。7500万パラメータ。
  96×96 の STL-10 モデルの流用。128×128 での attention（16,384 トークン）のせいで非常に遅く、メモリも大量に使った。
- チェックポイントには `"arch"` が保存される。これがないファイルは v1。**v1 と v2 に互換性はない。**
- Loss: 予測ノイズの MSE。Optimizer: AdamW（lr 1e-4、weight decay 1e-4）。
- **学習率スケジュール:** 1000 ステップの線形ウォームアップ（`--warmup-steps`）→ 最後のステップで 0 になるコサイン減衰。
- UNet の forward は **bf16 autocast**（loss、透かし loss、重みは fp32 のまま）。`--no-amp` でオフ。TF32 も有効。
- 学習用の `DDPMScheduler`: 1000 ステップ、beta は 1e-4 から 0.02 まで線形。
- **サンプリング:** 同じ DDPM 設定から作った DPM-Solver++（`make_sampling_scheduler`）、デフォルト 50 ステップ（`--sample-steps`）。
  条件付きと条件なしの予測を 1 回の forward でまとめて計算する。

## 3. EMA（`--ema-decay 0.9995`）

- 学習用の重み `w` とは別に、重みの指数移動平均 `w_ema = 0.9995 × w_ema + 0.0005 × w` を毎ステップ更新して持つ。
- 拡散モデルはランダムな時刻・ノイズ・バッチで学習するので、重みがステップごとに揺らぐ。平均をとると揺らぎが消えて、生成画像がきれいになる。
- 0.9995 → だいたい直近 2000 ステップ ≈ 30 エポック分の平均（新しいものほど重い）。
- 学習（勾配の更新）は `w` で行い、**サンプル画像と FID は EMA モデルで作る**。チェックポイントには両方保存（`model_state_dict`, `ema_state_dict`）。
- 透かしの loss と `watermarked_model.pth` は `w` を使う ❓ `lambda_>0` に戻すときは、EMA 重みからも透かしを取り出せるか確認する。

## 4. CFG

- **学習時**（`--cfg-dropout 0.1`）: 各ラベルを 10% の確率で空ラベル 2 に置き換える。これで 1 つのモデルが「条件付き」と「条件なし」の両方の予測をできるようになる。
- **生成時**（`--cfg`、デフォルト 3.0）: `eps = eps_uncond + s * (eps_cond - eps_uncond)`。
  `s = 0` → ラベルを無視、`s = 1` → 普通の条件付き、`s > 1` → クラスらしさを強調（大きすぎると色が飽和し、多様性が減る）。
- **`--cfg` は学習そのものには影響しない。** 途中のサンプル画像、最終 FID、ファイル名にだけ使う。学習後に `eval_fid.py --cfg` で変えて評価できる。

## 5. 透かし（`LSUNembedding.py` + `LSUNtrain.py`）

- 対象の層: **最後の up ブロックの最後の conv2**（`LSUNfunc.watermark_layer_path`）:
  v1 = `up_blocks.3.resnets.2.conv2`、v2 = `up_blocks.5.resnets.2.conv2`。どちらも 128→128、3×3 → 重み 147,456 個。
- 選ぶ重みの数 `wmark_m = 18000`、透かしのビット数 `wmark_dim = 4096`。
- 秘密鍵 `K`: `randn(18000, 4096)`、シード `w_key_seed=0`。透かしビット `b`: `randint(0,2,(1,4096))`、シード `wmark_seed=123`。
- **エポック 0（通常の学習だけ）:** 対象の層の `|grad|` を全バッチで足し合わせる。
  エポック 0 の最後に、足し合わせた勾配で重みを並べ替え、先頭から連続した `wmark_m` 個を埋め込み位置にする。
  - `SORT=False`（デフォルト）→ 昇順 → **勾配が最も小さい重み**
  - `SORT=True`（`--sort`）→ 降順 → **勾配が最も大きい重み**
  - 研究として両方を比べるための切り替え（ユーザー確認済み）。
  - `watermarked_model.pth` を保存 = {エポック 0 終了時点のモデル重み、インデックス、秘密鍵、層の名前}。ビット列はシードから作り直す。
- **エポック 1 以降:** loss = `MSE + lambda_ * BCE(sigmoid(w_selected · K), b)`。今は `lambda_ = 0`（意図的）。
- 取り出し・検出: `sigmoid(w_selected · K) > 0.5` が `b` と一致すればよい ❓ 検出スクリプト（`../detection*.py`）はまだ読んでいない。

## 6. 保存と評価

- 出力フォルダ: `content/runs/<日付>/`（または `--save-dir`）。
  - 毎エポック: loss グラフ、`checkpoint_latest.pt`（`--resume` 用）、`best_model.pt`（学習 loss が最小のもの。画質の指標ではない）。
  - `--save-every 50` エポックごと: `checkpoint_epNNNN.pt`。
  - `--sample-every 25` エポックごと: `samples/epoch_NNNN.png`（上の段 bedroom、下の段 cat、EMA 重み）。
- 最終 FID（`evaluate_and_calculate_fid`）: EMA モデルで `--fid-samples 2000` 枚（bedroom と cat を半分ずつ）生成し、
  `./fid_ref_lsun`（本物の画像 1000 枚すべて）と pytorch-fid で比較。生成画像は削除する。
  クラスごとに16枚を `raw_generate_final_check/` に保存。IS は計算しない（常に 0.0）。
- ハイパーパラメータ、学習時間、FID を `<save_dir>/training_log.txt` に追記する。
- **wandb**（`--wandb`）: 10 ステップごとの loss・学習率、エポックごとの平均 loss・時間、サンプル画像、最終 FID と学習時間。
  GPU 使用率・電力・温度は wandb が自動で記録する。`--resume` では同じ run に続けて記録する（run id をチェックポイントに保存）。

## 7. 速度（GB10、バッチ 16、GPU を単独で使ったとき）

| | 学習 1 ステップ | サンプリング 1 ステップ（CFG 付きで16枚） |
|---|---|---|
| v2 + bf16 | **1.9 秒** → 約 120〜130 秒/エポック → **600 エポック ≈ 22 時間** | 1.5 秒 |
| v2 fp32 | 2.5 秒 | 1.9 秒 |
| v1（旧）+ bf16 | 60 秒以上 | — |

- FID 用の生成: 25 ステップで **1 枚 ≈ 1.3 秒**（2000 枚 ≈ 43 分、2026-11-27 実測）。50 ステップならその約 2 倍。
- Claude 導入前との比較: 学習 1 エポック 約 63 分 → 約 3.6 分（約 17 倍）。生成 1 枚 約 19 分 → 約 1.3 秒。
  FID 評価 約 42 時間（約 100 枚）→ 約 45 分（2000 枚）。
- 別の学習と GPU を共有すると大きく遅くなる（2026-11-24 の学習は 22 時間の見込みが 36 時間かかった）。

## 8. メモリ（2026-11-27 実測）

- DGX Spark（GB10）は **CPU と GPU が同じメモリ（約 122 GiB）を共有**。あふれると CUDA のエラーではなく swap → **マシン全体がフリーズ**する。
- 学習 1 ステップのピーク（v2、bf16 + EMA、実測）: バッチ 16 = **31.1 GiB**（2.02 秒/ステップ）、バッチ 32 = **60.2 GiB**（3.90 秒/ステップ）。
  **1 枚あたり約 1.8 GiB**（≈ 1.5 + 1.83 × バッチ GiB）。64 ≈ 118 GiB（あふれる）。メモリを使うのは中間データ（activation）で、データセットの大きさは関係ない。
- v1（fp32）は 1 枚あたり約 3.5 GiB（バッチ 16 = 56.8 GiB、60 秒/ステップ）。
- Claude 導入前に「1 エポック目の前（ゲージが進まないまま）フリーズ」していた原因: **v1 モデルのメモリ**（バッチ 32 なら約 110 GiB）。
  詳細は `experiments.md` の「学習 1 ステップの測定」。（以前は TensorFlow のせいと書いたが、TF からはこの GPU が見えないので主因ではなかった。）
- `--max-mem-gb`（デフォルト 80）で PyTorch の使用量に上限をかけている（`LSUNfunc.limit_gpu_memory`）。超えたらフリーズせず `OutOfMemoryError` で止まる。
  別コンテナ（`yuka-var`）の使用量は制限できない。
- バッチを大きくしても、GPU がすでにフルに働いていれば 1 エポックの時間はほぼ同じ。更新回数が減るので学習率の調整も必要になる。

## 9. さらに速くする案（未実施）
- `torch.compile`、channels_last。

# CLAUDE.md: LSUN 透かし入り拡散モデル

このファイルは、Claude がこのプロジェクトとユーザーの指示をどう理解しているかを記録したものです。
**間違いがあればユーザーが確認して直してください。** ❓ の付いた項目は推測か、まだ答えが出ていない質問です。

---

## 1. ユーザーの指示（作業の進め方）

1. **コードの変更はすべて GitHub に記録する。** 修正ごとに小さなコミットを1つ作り、メッセージには「なぜ変えたか」を書く。
   そのあと `git@github.com:catallergyy/LSUN-diffusion.git`（ブランチ `main`）に push する。
   ユーザーは GitHub で変更を1つずつ確認する。
2. **変更の前に説明する。** まず計画を提案し、「はい」をもらってからコードを編集する。
3. **大きなファイルはコミットしない。** データ、画像、チェックポイント（`*.pt`, `*.pth`）、`*.npz` はローカルに置いたままにする（`.gitignore`）。
4. **ユーザーへの説明は日本語で行う**（2026-11-27 に変更。以前は簡単な英語）。コードのコメントも日本語。
5. 現在の優先事項:
   - **生成画像をもっときれいにする**（FID を下げる）。今は透かしをオフにしている（`lambda_=0`）。
   - **評価と学習に時間がかかりすぎる**（FID 評価に約40時間かかっていた）。研究の中身を壊さずに速くする。

---

## 2. プロジェクトの概要

2クラス（**bedroom** と **cat**）を生成する、**クラス条件付き拡散モデル**（DDPM、256×256）。特徴は次のとおり。

- **Classifier-Free Guidance（CFG）**: 学習時にラベルをランダムに消し（ラベルドロップアウト）、生成時にガイダンススケールをかける。
- **モデルの重みに埋め込む透かし**: 秘密のビット列を、ある1つの層の選んだ重みに隠す。
  目的はモデルの所有権を証明すること。
- **FID**（pytorch-fid）: 透かしを入れても画質が落ちていないかを確認する。

Claude の理解している研究テーマ: *FID を透かしなしのモデルに近いまま保ちながら、透かしを確実に埋め込み・取り出せるか？* ❓

同じアイデアの古い STL-10 版は `../`（ルートのスクリプト）と `../STL/` にある。

---

## 3. ファイル

| ファイル | 役割 |
|---|---|
| `LSUNdataloader.py` | Hugging Face から画像を `LSUN_origin/{bedroom,cat}` にダウンロードし、256×256 にリサイズ・中央切り抜きして `dataset/val_256/{bedroom,cat}` に保存する（各クラス500枚、合計1000枚）。 |
| `LSUNfunc.py` | モデルとスケジューラの作成、データ読み込み、CFG サンプリング、FID 参照フォルダの作成、FID 評価。 |
| `LSUNembedding.py` | 透かしの設定と、鍵・ビット列の生成関数。 |
| `LSUNtrain.py` | メインスクリプト: 学習 → 透かし埋め込み → 保存 → FID 計算 → ログ書き出し。 |
| `eval_fid.py` | （Claude が追加）学習せずに、保存済みチェックポイントから FID だけを計算する。 |
| `fid_test.py` | 古い FID テスト（TensorFlow 版）❓ たぶんもう使っていない。 |
| `fid_stats_lsun_{bedroom,cat}.npz` | 計算済みの FID 統計量 ❓ 出どころと Inception のバージョンは不明。コードでは使われていない。 |
| `content/runs/<日付>/` | 出力: チェックポイント、loss のグラフ、サンプル画像、`training_log.txt`。 |
| `archive/` | 古い資料（9.5 GB、git 管理外）。 |

### 実行方法（devcontainer `charming_tu` の中、パスは `/workspaces/research/diffusion-model/LSUN`）
ホスト側の Python には `diffusers` が入っていないので、必ずコンテナの中で実行する。例:
`docker exec -w /workspaces/research/diffusion-model/LSUN charming_tu python3 ...`
GPU: NVIDIA GB10（PyTorch が「capability 12.1 > 12.0」と警告を出すが、動作する）。

```
python3 LSUNtrain.py --epochs 600                         # 通常の学習（lambda_=0）
python3 LSUNtrain.py --epochs 600 --sort                  # SORT=True（勾配が最も大きい重みを選ぶ）
python3 LSUNtrain.py --epochs 800 --resume content/runs/<日付>/checkpoint_latest.pt   # 学習の続き
python3 eval_fid.py content/runs/<日付>/checkpoint_ep0600.pt --steps 25 --cfg 2.0     # FID だけ計算
python3 LSUNtrain.py --help                               # 全オプション
```

---

## 4. 処理の詳細

### 4.1 データ
- `DatasetFolder("./dataset/val_256")` → クラスはアルファベット順: **bedroom = 0, cat = 1**。
- [-1, 1] に正規化。バッチサイズ 16（`--batch-size`）→ 1エポック 63 バッチ。
- bedroom 画像は `pcuenq/lsun-bedrooms` から。**cat 画像はユーザーが Web から集めた本物の LSUN cat 画像**
  （ユーザー確認済み）。データは問題ないので変更しない。
  （`download_lsun_samples_from_hf` には `cats_vs_dogs` のダウンロードが残っているが、今の cat データはそこから来たものではない。）
- データセットは小さい（1000枚）。学習と FID の参照に同じ画像を使っている。

### 4.2 モデル（`LSUNfunc.create_model(lr, arch)`）
- diffusers の `UNet2DModel`、256×256、`num_class_embeds=3`（0 = bedroom, 1 = cat, **2 = CFG 用の空ラベル**）、各ブロックに ResNet 層 2つ。
- **`v2`（デフォルト、新）:** 6段階 256→128→64→32→16→8、チャネル (128,128,256,256,512,512)、
  self-attention は **16×16 だけ**（+ 中間ブロック 8×8）、head dim 64。1億1400万パラメータ。DDPM 論文の LSUN-256 モデルに近い。
- **`v1`（旧、古いチェックポイントを読むために残している）:** 4段階、attention は 128/64/32 px、head dim 8。7500万パラメータ。
  96×96 の STL-10 モデルをそのまま流用したもの。128×128 での attention（16,384 トークン）のせいで非常に遅かった。
- チェックポイントには `"arch"` が保存される。これがないファイルは v1。**v1 と v2 のチェックポイントに互換性はない。**
- Loss: 予測ノイズの MSE。Optimizer: AdamW（lr 1e-4、weight decay 1e-4）。
- **学習率スケジュール:** 1000 ステップの線形ウォームアップ（`--warmup-steps`）→ `--epochs` の最後のステップで 0 になるコサイン減衰。
- **EMA 重み**（`--ema-decay 0.9995`）: 重みの移動平均。**サンプル画像と FID は EMA モデルで作る。**
  透かしの loss と `watermarked_model.pth` は生の（EMA でない）重みを使う ❓ lambda_>0 に戻すときは、
  検出を EMA 重みと生の重みのどちらで行うべきか確認する。
- UNet の forward には **bf16 autocast** を使う（loss、透かし loss、重みは fp32 のまま）。`--no-amp` でオフにできる。
- 学習用の `DDPMScheduler`: 1000 ステップ、beta は 1e-4 から 0.02 まで線形。
- **サンプリング:** 同じ DDPM 設定から作った DPM-Solver++（`make_sampling_scheduler`）、デフォルト 50 ステップ（`--sample-steps`）。

### 4.3 CFG
- 学習時: 各ラベルを確率 `cfg_dropout_prob = 0.1` で空ラベル 2 に置き換える。
- 生成時: `eps = eps_uncond + s * (eps_cond - eps_uncond)`、`s = guidance_scale = 3.0`。

### 4.4 透かし（`LSUNembedding.py` + `LSUNtrain.py`）
- 対象の層: **最後の up ブロックの最後の conv2**（`LSUNfunc.watermark_layer_path`）:
  v1 = `up_blocks.3.resnets.2.conv2`、v2 = `up_blocks.5.resnets.2.conv2`。どちらも 128→128、3×3 → 重み 147,456 個。
- 選ぶ重みの数 `wmark_m = 18000`、透かしのビット数 `wmark_dim = 4096`。
- 秘密鍵 `K`: `randn(18000, 4096)`、シード `w_key_seed=0`。
  透かしビット `b`: `randint(0,2,(1,4096))`、シード `wmark_seed=123`。
- **エポック 0（通常の学習だけ）:** 対象の層の `|grad|` を全バッチで足し合わせる。
  エポック 0 の最後に、足し合わせた勾配で重みを並べ替え、先頭の `wmark_m` 個を埋め込み位置にする。
  - `SORT`（`--sort` フラグ）は**実験用の切り替え**（ユーザー確認済み）。選んだ重みの勾配の大きさが
    透かしの埋め込みにどう影響するかを比べる研究:
    - `SORT=False` → `torch.sort(descending=False)` → **勾配が最も小さい重み**（現在の設定、意図どおり）
    - `SORT=True` → **勾配が最も大きい重み**
    このオプションは残すこと。どちらか一方に固定しない。
  - コードは先頭から連続した `wmark_m` 個の重みを取る（「奇数番目のインデックス」「容量2倍」という古いコメントは実装されていなかったので削除した）。
  - `watermarked_model.pth` を保存する = {*エポック 0 終了時点*のモデル重み、インデックス、秘密鍵、層の名前}。
    ビット列そのものは保存しない。シードから作り直す。
- **エポック 1 以降:** loss = `MSE + lambda_ * BCE(sigmoid(w_selected · K), b)`。
- 取り出し・検出: `sigmoid(w_selected · K) > 0.5` が `b` と一致すればよい ❓ 検出スクリプト（`../detection*.py`）はまだ読んでいない。
- `lambda_ = 0` は**意図的**（ユーザー確認済み）: **今の目標は生成画像をきれいにすること**なので、
  透かしは今はオフ。頼まれない限り `lambda_` は変えない。
- エポック数は**自由に変えられること**: `--epochs N`（デフォルト 600）。これまでの実験は **600 エポック**。

### 4.5 保存と評価
- 出力フォルダ: `content/runs/<日付>/`（または `--save-dir`）。毎エポック: loss のグラフ、`checkpoint_latest.pt`（`--resume` 用）、
  `best_model.pt`（学習 loss が最小のもの）。`--save-every 50` エポックごと: `checkpoint_epNNNN.pt`。
  `--sample-every 25` エポックごと: `samples/epoch_NNNN.png`（上の段が bedroom、下の段が cat、EMA 重み）→ **画質が良くなっていくかはこれを見る**。
- ⚠️ ディスク: チェックポイント1つ ≈ 1.8 GB（モデル + EMA + optimizer）。600 エポックを `--save-every 50` で回すと ≈ 25 GB。
- 最終 FID（`evaluate_and_calculate_fid`）: EMA モデルで `--fid-samples 2000` 枚（bedroom と cat をちょうど半分ずつ）生成し、
  `./fid_ref_lsun`（本物の画像 1000 枚すべて）と pytorch-fid で比較し、生成画像は削除する。
  クラスごとに16枚のサンプルを `raw_generate_final_check/` にも保存する。IS は計算しない（常に 0.0）。
- ハイパーパラメータ、学習時間、FID を `<save_dir>/training_log.txt` に記録する。
- 参照画像が1000枚の FID はばらつきが大きい。実験同士の比較には使えるが、論文の数値とは比べられない。

### 4.6 実測の速度（GB10、バッチ 16、2026-11-24）
| | 学習 1 ステップ | サンプリング 1 ステップ（CFG 付きで16枚） |
|---|---|---|
| v2 + bf16 | **1.9 秒** → 約 130 秒/エポック → **600 エポック ≈ 22 時間** | 1.5 秒 |
| v2 fp32 | 2.5 秒 | 1.9 秒 |
| v1（旧）+ bf16 | 60 秒以上（計測を途中で止めた） | — |

- 2000枚 × 50 ステップの最終 FID ≈ 2.6 時間。`--steps 25`（または `--sample-steps 25`）なら ≈ 1.3 時間。
- 以前の評価（v1、DDPM 1000 ステップ、1ステップに forward 2回）が約40時間かかった理由はこれ。

## 5. Claude が行った変更（GitHub のコミットを参照）

| コミット | 変更内容 |
|---|---|
| `ac23bbd` | **バグ修正:** STL-10 用のラベル変換（`labels==9 → 1`）を削除。これのせいで cat がすべてラベル 0（bedroom）になり、cat クラスが学習されていなかった。**この修正より前のチェックポイントはすべてこのバグを含む。** |
| `1382d64` | **バグ修正:** FID の参照画像がデータセットの先頭100枚で、すべて bedroom だった。今は両クラスの1000枚すべて（`./fid_ref_lsun`）。 |
| `c06d6f8` | TensorFlow、使っていない InceptionV3 モデル2つ、使っていないコードを削除。DataLoader `num_workers=4`。 |
| `93e1e54` | サンプリング: DDPM 1000 ステップをやめて DPM-Solver++ 50 ステップに、cond と uncond を1回の forward で計算、bf16。FID の生成枚数 100 → 2000（100枚では FID として不十分）。 |
| `ca510d5` | チェックポイントを評価する `eval_fid.py` を追加。 |
| `57bc28d` | 新しい UNet 構造 **v2**（attention は 16×16 だけ）。古いチェックポイント用に `v1` も残す。 |
| `b37958e` | `LSUNtrain.py` の整理: コマンドラインオプション（`--epochs` など）、チェックポイント、サンプル画像、`--resume`。2つあった学習ループを1つにまとめた。 |
| `a0e6a4a` | **EMA** 重み。サンプルと FID は EMA を使う。 |
| `2e905d7` | **学習率のウォームアップ + コサイン減衰**。 |
| `c6c8023` | 学習に **bf16 autocast** + TF32。 |

✅ 2026-11-24 に devcontainer でテスト済み: 2エポックの実行で、学習 → 透かし位置の選択 →
チェックポイント → サンプル画像 → FID → ログ まで、エラーなく通った。

### 最初の長い学習: `content/runs/2026-11-24/`（v2、600 エポック、lambda_=0、CFG 3.0）
- 完了。**FID = 144.2**（EMA、2000枚、DPM-Solver++ 50 ステップ）。学習時間 **35時間54分**（見積もりは約22時間）。学習後の FID も 5.5 時間かかった（見積もり 2.6 時間）。
  原因はたぶん **同じ GPU での別の学習との競合**（ユーザーの判断）。学習中の 11/24 18:16 に、別のコンテナ `yuka-var` で tmux セッション `wm0b` が始まっていた。
  単独で動かしたときの実測: 生成 1 枚 ≈ 1.3 秒（25 ステップ）。時間を比べるときは、GPU を他の処理と共有していないか確認すること。
- `samples/epoch_0600.png`: bedroom は部屋だとわかるが、色が強く偏っていて（全体がピンク、全体が青など）、質感がざらざらしている。
  cat はほとんど失敗（色のかたまりで、毛や顔の形がわかるものは少しだけ）。
  色が濃すぎるのは CFG スケール 3.0 が大きすぎるためかもしれない ❓。

---

## 6. 次のステップ（提案、まだ未実施）

1. ~~ユーザーが本番の学習を始める~~ → 完了（上記）。
2. **（実行中、2026-11-27 開始）** `2026-11-24/checkpoint_ep0600.pt` で、`eval_fid.py --cfg` を使って FID が最も良い CFG スケールを探す（例: 1.5 / 2 / 3 / 5）。
3. 「ベストモデル」は学習 loss ではなく、FID か最後の EMA チェックポイントで選ぶ（拡散モデルの loss は画質とほとんど関係しない）。
4. さらに速くする案（未実施）: `torch.compile`、channels_last。

## 7. ユーザーへの質問 ❓
- ~~*勾配が最も小さい*重みを選ぶ（`SORT=False`）のは意図どおりか？~~ → **はい。** `SORT` は勾配の大小を比べるための実験オプション（2026-11-24 回答）。
- ~~`lambda_=0` / 1エポックはベースライン実験か、間違いか？~~ → **意図的。**（2026-11-24 回答）
- ~~`cats_vs_dogs` を「cat」クラスとして使ってよいか？~~ → **cat データはユーザーが集めた本物の LSUN cat 画像。問題なし**（2026-11-24 回答）。
- `fid_stats_lsun_*.npz` はどこから来たものか？

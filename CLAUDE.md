# CLAUDE.md: LSUN 透かし入り拡散モデル

このファイルは、Claude がこのプロジェクトとユーザーの指示をどう理解しているかを記録したもの。会話のたびに毎回読み込まれるので、**短く保つ**。
**間違いがあればユーザーが確認して直す。** ❓ の付いた項目は推測か、まだ答えが出ていない質問。

詳しい内容は次のファイルにある（必要なときに読む）:
- [`docs/experiments.md`](docs/experiments.md): **実験ノート**。実験ごとの設定・FID・時間・画像の印象・考察。実験が終わるたびに追記する。
- [`docs/pipeline.md`](docs/pipeline.md): **仕組みの詳しい説明**。データ、モデル構造、CFG、EMA、透かし、保存と評価、速度・メモリの実測値。
- 変更の履歴は **GitHub のコミット**（`git log`）を見る。

---

## 1. ユーザーの指示（作業の進め方）

1. **コードの変更はすべて GitHub に記録する。** 修正ごとに小さなコミットを1つ作り、メッセージには「なぜ変えたか」を書く。
   そのあと `git@github.com:catallergyy/LSUN-diffusion.git`（ブランチ `main`）に push する（言われなくても適宜 push してよい）。
   ユーザーは GitHub で変更を1つずつ確認する。
2. **変更の前に説明する。** まず計画を提案し、「はい」をもらってからコードを編集する。
3. **大きなファイルはコミットしない。** データ、画像、チェックポイント（`*.pt`, `*.pth`）、`*.npz`、`wandb/` はローカルに置いたままにする（`.gitignore`）。
4. **ユーザーへの説明は日本語で行う**（2026-11-27 に変更。以前は簡単な英語）。コードのコメントも日本語。
5. 現在の優先事項:
   - **生成画像をもっときれいにする**（FID を下げる）。今は透かしをオフにしている（`lambda_=0`）。
   - **評価と学習の時間を短くする**（研究の中身を壊さずに）。
6. 実験が終わったら、結果を `docs/experiments.md` に追記する。
7. **時刻は日本時間（JST）で書く**（2026-11-27 に追加）。サーバーの時計は UTC なので、ログやファイルの時刻は +9 時間して伝える。
   ⚠️ これより前に書いた docs の時刻（`docs/experiments.md` など）は UTC のまま。

---

## 2. プロジェクトの概要

2クラス（**bedroom** と **cat**）を生成する、**クラス条件付き拡散モデル**（DDPM、256×256）。
- **Classifier-Free Guidance（CFG）**: 学習時にラベルをランダムに消し、生成時にガイダンススケールをかける。
- **モデルの重みに埋め込む透かし**: 秘密のビット列を、ある1つの層の選んだ重みに隠す。目的はモデルの所有権の証明。
- **FID**（pytorch-fid）: 透かしを入れても画質が落ちていないかを確認する。

研究テーマ（Claude の理解）: *FID を透かしなしのモデルに近いまま保ちながら、透かしを確実に埋め込み・取り出せるか？* ❓

同じアイデアの古い STL-10 版は `../`（ルートのスクリプト）と `../STL/` にある。

---

## 3. ファイル

| ファイル | 役割 |
|---|---|
| `LSUNdataloader.py` | Hugging Face から画像をダウンロードし、256×256 にして `dataset/val_256/{bedroom,cat}` に保存（各クラス500枚）。 |
| `LSUNfunc.py` | モデル作成、データ読み込み、CFG サンプリング、FID 参照フォルダ作成、FID 評価、GPU メモリ上限。 |
| `LSUNembedding.py` | 透かしの設定と、鍵・ビット列の生成関数。 |
| `LSUNtrain.py` | メイン: 学習 → 透かし位置の選択 → 保存 → FID → ログ。 |
| `eval_fid.py` | 学習せずに、チェックポイントから FID だけを計算する。 |
| `fid_test.py` | 古い FID テスト（TensorFlow 版）❓ たぶんもう使っていない。 |
| `fid_stats_lsun_{bedroom,cat}.npz` | 計算済みの FID 統計量 ❓ 出どころ不明。コードでは使われていない。 |
| `content/runs/<日付>/` | 出力: チェックポイント、loss グラフ、`samples/`、`training_log.txt`。 |
| `archive/` | 古い資料（9.5 GB、git 管理外）。 |

## 4. 実行方法

**必ず devcontainer `charming_tu` の中で実行する**（ホストの Python には `diffusers` がない）:
`docker exec -w /workspaces/research/diffusion-model/LSUN charming_tu python3 ...`

```
python3 LSUNtrain.py --epochs 600 --wandb                 # 通常の学習（lambda_=0、左右反転あり、wandb に記録）
python3 LSUNtrain.py --epochs 600 --sort                  # SORT=True（勾配が最も大きい重みを選ぶ）
python3 LSUNtrain.py --epochs 600 --no-hflip              # 左右反転なし（2026-11-24 の実験と同じ条件）
python3 LSUNtrain.py --epochs 800 --resume content/runs/<日付>/checkpoint_latest.pt   # 学習の続き
python3 eval_fid.py content/runs/<日付>/checkpoint_ep0600.pt --steps 25 --cfg 2.0     # FID だけ計算
python3 LSUNtrain.py --help                               # 全オプション
```

- **git push はホストから**行う。`SSH_AUTH_SOCK=~/.ssh/agent.sock` が必要。鍵は `~/.ssh/id_ed25519_github`。
  マシンを再起動したら、ユーザーが普通のターミナルで `ssh-agent -a ~/.ssh/agent.sock` と `ssh-add ~/.ssh/id_ed25519_github` を実行する必要がある。
- **wandb** はコンテナ内でログイン済み（エンティティ `m1m02530142-n`、プロジェクト `LSUN-diffusion`）。

### 接続が切れたときのために（tmux）

ユーザーは NVIDIA Sync（SSH）で接続している。SSH が切れると、その下で動いていたもの（Claude を含む）は終了する。
（2026-11-27 15:51 JST に実際に起きた。原因はたぶん、長い間通信がない接続をネットワーク機器が切ったこと。自動スリープはオフだった）

- **Claude は tmux の中で起動する:**
  ```
  tmux new -s claude        # 初回: tmux のセッションを作る
  claude --continue         # 前の会話の続きから起動する
  ```
- **切れた後の再接続:** Sync で接続し直して `tmux attach -t claude`（Claude も会話もそのまま残っている）。
- tmux から一時的に抜ける: `Ctrl+b` → `d`。セッションの一覧: `tmux ls`。
- **数十分以上かかる処理（学習・FID）も、Claude が tmux の別セッションで起動する**（例: `tmux new -d -s cfgsweep <スクリプト>`）。
  Claude の Bash のバックグラウンド実行だけだと、Claude が終了したときに一緒に止まる。
- 予防: PC 側の `~/.ssh/config` に `ServerAliveInterval 60` を書く（Sync がこの設定を読むかどうかは未確認 ❓）。

---

## 5. 絶対に注意すること ⚠️

- **ラベルのバグ:** commit `ac23bbd` より前のチェックポイントは、cat がすべて bedroom（ラベル 0）として学習されている。使わない。
- **v1 と v2 のチェックポイントに互換性はない。** `"arch"` がないファイルは v1。
- **`lambda_ = 0` は意図的。** 頼まれない限り変えない。
- **`SORT`（`--sort`）は実験用の切り替え**（勾配の小さい重み vs 大きい重み）。どちらか一方に固定しない。
- **データは変更しない。** cat 画像はユーザーが集めた本物の LSUN cat 画像。
- **メモリ:** DGX Spark は CPU と GPU が同じ約 122 GiB を共有していて、あふれるとマシン全体がフリーズする。
  バッチ 1 枚あたり約 1.8 GiB（バッチ 16 ≈ 31 GiB）。`--max-mem-gb`（デフォルト 80）で上限をかけている。
- **GPU の共有:** 別のコンテナ `yuka-var` も同じ GPU を使うことがある。学習を始める前と、時間を比べるときは `nvidia-smi` で確認する。
- **FID の参照画像は学習データそのもの**（1000枚）。実験同士の比較には使えるが、丸暗記（過学習）は検出できず、論文の数値とも比べられない。
- **「ベストモデル」は学習 loss ではなく FID で選ぶ**（拡散モデルの loss は画質とほとんど関係しない）。
- ディスク: チェックポイント1つ ≈ 1.8〜2.1 GB。600 エポックで約 25 GB。

---

## 6. ユーザーへの質問 ❓
- `fid_stats_lsun_*.npz` はどこから来たものか？
- 透かしを入れるとき（`lambda_>0`）、検出は EMA 重みと生の重みのどちらで行うべきか？
- 検出スクリプト（`../detection*.py`）はまだ読んでいない。

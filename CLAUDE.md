# CLAUDE.md

> このファイルはプロジェクトの進行に合わせて Claude Code 自身が更新していく「生きたドキュメント」です。
> 新規プロジェクト開始時はほぼ空の状態で構いません。実装が進むにつれて、下記のルールに従って自動的に育てていってください。

---

## Claude Codeへの指示（自己更新ルール）

あなたはこのプロジェクトで作業するたびに、以下のルールに従って本ファイル（CLAUDE.md）を更新してください。

### 更新すべきタイミング
- 新しいファイル/モジュール/ディレクトリを作成したとき
- 既存のアーキテクチャや設計方針を変更したとき（例: ライブラリの乗り換え、レイヤー構成の変更）
- 「今後も同じ説明をユーザーにさせそうだ」と感じたとき
- ユーザーから明示的に「CLAUDE.mdに書いて」と言われたとき

### 更新時の振る舞い
- 該当する章（下記「## 構成」参照）に、簡潔な箇条書きで追記する。長文の説明文は書かない。
- 「Critical Paths（ファイル所在表）」は特に優先して最新化する。何かを追加/変更したら、まずここを疑う。
- 迷ったら、ユーザーに聞かずにいったん追記し、後で「CLAUDE.mdのここを追記しました」と一言報告する。
- 冗長な履歴は残さない。古い情報は書き換える（変更履歴セクション以外は追記ではなく上書き優先）。
- コード規約・設計判断の「理由」も一言添える（例: 「認証は express-session を採用（JWT不要な単一サーバー構成のため）」）。

### 書かないこと
- 実装の詳細なロジック説明（コード自体やdocstringに書くべきもの）
- 一時的なタスクの進捗（TODOリストや作業ログはここに書かない）
- 憶測・未確定の設計（決まったことだけを書く）

---

## 構成

### 1. プロジェクト概要

- 目的: GGUF形式の拡散モデル／テキストエンコーダ／ビジョンモデルを ComfyUI で読み込むカスタムノード。チェックポイント→GGUF変換も持つ。
- 系譜: city96/ComfyUI-GGUF → molbal/ComfyUI-GGUF（Q8_CR/Q4_CR、Dynamic VRAM、Krea 2 / Ideogram 4 等対応）→ 本フォーク Yoshiaki21/ComfyUI-GGUF。
- 想定ユーザー: 本人（AMD Radeon / ROCm 環境の ComfyUI 利用者）。

### 2. 技術スタック

- 言語/フレームワーク: Python 3.12、PyTorch（ROCm版）、ComfyUI 0.39.2 のカスタムノード
- DB: なし
- 主要ライブラリ: `gguf>=0.13.0`、`comfy_kitchen`（Q8_CR/Q4_CR と Dynamic VRAM のレイアウト）、Triton（任意。K系融合カーネル）

### 3. ディレクトリ構造 / Critical Paths（最重要）

| やりたいこと | 場所 |
|---|---|
| GGML ブロック展開（PyTorch実装、K系含む） | `dequant.py` |
| 実行時の重みラッパー・on-the-fly 展開（通常経路） | `ops.py`（`GGMLLayer` / `GGMLOps.Linear`） |
| Dynamic VRAM 経路の展開・Linear ディスパッチ | `quant_ops.py`（`GGMLLayout`） |
| K系 Linear のネイティブバックエンド選択・フォールバック | `kquant_backend.py`（`try_kquant_linear`） |
| K系 Triton 融合 matmul カーネル | `kquant_triton_backend.py` |
| 展開／matmul／K系カーネルの所要時間計測（切り分け用、既定無効） | `dequant_profile.py`（`COMFYUI_GGUF_DEQUANT_PROFILE=1`） |
| GGUF 読み込み・メタデータ・アーキ判定 | `loader.py` |
| ComfyUI ノード定義 | `nodes.py` |
| LoRA 適用 | `lora.py` |
| 変換ツール | `tools/convert.py` |
| テスト | `tests/`（実行方法は「よく使うコマンド」） |
| K系展開の参照実装一致テスト／ROCm ゲートのテスト | `tests/test_kquant_dequant.py` |
| 設計資料 | `ARCHITECTURE.md`, `AGENTS.md`, `docs/k-quant-inference.md` |
| 参照用 leejet 版（読むだけ。必要時に clone） | https://github.com/leejet/ComfyUI-GGUF（`dequant.py`/`ops.py` は city96 版と同一） |
| 参照用 city96 版（差分の基準。必要時に clone） | https://github.com/city96/ComfyUI-GGUF |

### 4. 設計方針・規約

- 標準GGML量子化は「展開してから PyTorch で計算」が基本。ネイティブ低ビット実行は Q8_CR（INT8）と、K系の任意 Triton バックエンドのみ。
- 新しい量子化型は `dequant.py` と `quant_ops.py` の両方で対応する（通常経路と Dynamic VRAM 経路の挙動を揃えるため）。
- 「ロード時に全重みを展開して保持」を既定にしない（GGUF の VRAM 節約の利点が消えるため）。
- 本フォークでの修正箇所には日本語コメント（何を・なぜ）。leejet/city96 から移植したコードは出典をコメントに残す。
- LICENSE と既存の著作権表記は変更しない。

### 5. よく使うコマンド

```bash
# テスト：ホストには torch がないため、同じイメージの使い捨てコンテナで実行する（稼働中コンテナには触れない）
# GPU テストも走らせる場合は --device /dev/kfd --device /dev/dri を付け、HIP_VISIBLE_DEVICES で 8060S を選ぶ
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
  -v $PWD:/work:ro -v /home/lucifar/data/comfyui/app-rocm10:/app:ro -v /home/lucifar/data/comfyui/venv-rocm10:/opt/venv:ro \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/work -w /work --entrypoint /bin/bash registry.gskraft.com/comfyui-rocm10 \
  -c 'export HIP_VISIBLE_DEVICES=$(/opt/venv/bin/python3 /opt/get_gpu_index.py 8060S); /opt/venv/bin/python3 -m unittest discover -s tests -t .'

# ComfyUI のログ確認（読み取りのみ）
journalctl --user -u comfyui-igpu --no-pager | grep -i gguf

# ComfyUI 再起動はユーザーが行う（Claude は実行しない）
# systemctl --user restart comfyui-igpu
```

### 6. 既知の制約・注意点

- 本リポジトリはコンテナ `comfyui-igpu` の `/app/custom_nodes/ComfyUI-GGUF` にバインドマウントされている。編集は ComfyUI 再起動後に即反映される。
- ROCm 版 PyTorch でも `device.type == "cuda"` になる。CUDA 前提の分岐が AMD GPU でも有効になる点に注意。
- 同梱の K系 Triton カーネルは ROCm では既定で無効（拡散モデルで展開→rocBLAS より約4倍遅かったため）。有効化は `COMFYUI_GGUF_KQUANT_TRITON_ROCM=1`。
- `test_targeted_quantization` の Q4_CR / MiniMax H3 関連 5件は main の時点で失敗している（今回の変更とは無関係）。
- ComfyUI を import する既存テスト（`test_kquant_backend` など）は、GPU が見えない環境では ComfyUI 側の初期化で失敗する。
- 実機: Radeon 8060S（gfx1151、iGPU）、ROCm 10.0。今後 gfx1201（RX 9070 XT / R9700）も使用予定。
- モデル置き場: `/home/lucifar/data/comfyui/models/`（削除・移動しない）。
- `main` へ直接コミット・プッシュしない。作業ブランチを切る。

### 7. 変更履歴（任意）

- 2026-10-11: ROCm で同梱 K系 Triton カーネルを既定無効化。Qwen-Image 2.1 Q6_K が 24.70 → 5.26 s/it（8060S 実測）。

---

**運用メモ（ユーザー向け）**
- ある程度育ってきたら、章ごとに `docs/` 配下へ分割してこのファイルからリンクする構成に移行することを検討してください（tududiのCLAUDE.mdのような形）。
- 定期的に「CLAUDE.mdを見直して、実態と合っていない箇所を修正して」とClaude Codeに依頼すると、陳腐化を防げます。

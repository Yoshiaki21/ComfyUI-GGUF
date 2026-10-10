# Claude Code 指示書：K系量子化（Q4_K〜Q6_K）逆量子化の高速化

## 1. 目的

`Yoshiaki21/ComfyUI-GGUF`（molbal/ComfyUI-GGUF のフォーク）で、
**拡散モデルのK系量子化GGUFの推論速度を leejet/ComfyUI-GGUF と同等にする。**

molbal版のREADMEには「拡散モデルのK系は非対応。ロードできても推論が非常に遅い可能性がある」
と明記されており、実測でもその通りになっている。

## 2. リポジトリ

| 役割 | リポジトリ | 扱い |
|---|---|---|
| 作業対象 | https://github.com/Yoshiaki21/ComfyUI-GGUF | ここを修正する |
| 参照元（K系実装） | https://github.com/leejet/ComfyUI-GGUF | 読むだけ。変更しない |
| 共通の親 | https://github.com/city96/ComfyUI-GGUF | 差分の基準として参照可 |

準備（未実施なら行う）:

```bash
git clone https://github.com/Yoshiaki21/ComfyUI-GGUF
git clone https://github.com/leejet/ComfyUI-GGUF ../ref-leejet   # 参照用。作業対象の外に置く
cd ComfyUI-GGUF
git checkout -b feature/k-quant-dequant
```

## 3. 実測値（基準）

条件：Qwen-Image 2.1（7B）、画像編集ワークフロー、50ステップ、
Radeon 8060S（gfx1151）、ROCm 10.0、ComfyUI 0.39.2、
サンプラー・スケジューラ・CFG・解像度はすべて同一。

| ローダー | 形式 | 速度 | 備考 |
|---|---|---|---|
| なし | safetensors | 5.13 s/it | 基準 |
| molbal版 | Q8_0 | 5.50 s/it | 正常 |
| leejet版 | Q6_K | 6.35 s/it | **目標値** |
| molbal版 | Q6_K | 24.70 s/it | **これを直す** |

補足：
- 遅い実行中もGPU使用率は100%だった（CPUへのフォールバックではない可能性が高い）。
- VRAMは十分にあり、メモリ退避は起きていない。
- 同じROCm環境でleejet版は速いので、ROCm固有の問題ではなく実装差である。

## 4. 完了条件

1. molbal版ベースのQ6_Kが **6.35 s/it 前後**（safetensors比 +30%以内）になる。
2. Q8_0 の速度が劣化しない（5.50 s/it 前後を維持）。
3. 既存機能を壊さない：Krea 2 / Ideogram 4 の読み込み、Q8_CR / Q4_CR、
   メモリマップ読み込み、テキストエンコーダのGGUF読み込み、LoRA適用。
4. K系の展開結果が `gguf.quants.dequantize`（numpy参照実装）と数値的に一致する。

## 5. 進め方（必ずこの順序で。フェーズごとに報告して承認を待つ）

### フェーズ1：調査のみ（コードは変更しない）

- `dequant.py` / `ops.py` / `loader.py` / `quant_ops.py` / `nodes.py` について、
  作業対象と `../ref-leejet` の差分を整理する。
- `ARCHITECTURE.md` / `DESIGN.md` / `AGENTS.md` を読み、molbal版の設計方針
  （メモリマップ読み込み、Q8_CR/Q4_CR）を把握する。
- **molbal版でK系が遅くなる原因を特定する。** 想定される候補：
  - K系のtorch実装が削除されている／呼ばれていない
  - 展開が毎回別デバイスや別dtypeを経由している
  - メモリマップ経路でK系のテンソルが毎ステップ再読込・再転送されている
  - leejet版にある最適化（キャッシュ、in-place処理など）が入っていない
- 報告内容：原因、根拠となるコード箇所（ファイル名と行）、修正方針の案（複数あれば比較）。

→ **ここで停止し、私の承認を待つ。**

### フェーズ2：実装

- 変更は最小限にする。K系の展開経路以外には手を入れない。
- leejet版からコードを持ってくる場合は、どの関数をどこから移植したかをコメントに残す。
- 修正箇所には日本語コメントを入れる（何を・なぜ変えたか）。
- LICENSEと既存の著作権表記は変更しない。

### フェーズ3：テスト

- `tests/` に、K系（Q4_K, Q5_K, Q6_K。可能ならQ2_K, Q3_Kも）の展開結果を
  `gguf.quants.dequantize` と比較するテストを追加する。
  - ランダムなテンソルを量子化→展開し、参照実装との最大誤差を確認する。
  - CPUで実行できるようにする（GPUがなくても通ること）。
- 既存のテストがすべて通ることを確認する。

### フェーズ4：実機計測の手順を提示

速度計測は私が実機で行う。以下を示すこと：
- 修正版を ComfyUI の `custom_nodes` に反映する手順
- 確認すべきログ出力（フォールバック警告など）
- 原因切り分け用に、展開処理の所要時間を出すデバッグ手段
  （環境変数で有効化できる計測ログなど。既定では無効にする）

## 6. 実機で私が確認する組み合わせ

| モデル | 形式 | 確認内容 |
|---|---|---|
| Qwen-Image 2.1 | Q6_K | 速度が 6.35 s/it 前後になること |
| Qwen-Image 2.1 | Q8_0 | 5.50 s/it 前後を維持すること |
| Krea 2 | 既存のGGUF | 従来どおり読み込み・生成できること |
| いずれか | — | 修正前後で同一シードの出力画像が一致（またはほぼ一致）すること |

## 7. 環境情報

- OS：Manjaro Linux
- GPU：Radeon 8060S（gfx1151、iGPU）。今後 RX 9070 XT / Radeon AI PRO R9700（gfx1201）でも使用予定
- ROCm：10.0
- ComfyUI：0.39.2（podman / Quadlet のコンテナで稼働）
- 拡散モデルの保存先：`/home/lucifar/data/comfyui/models/diffusion_models/`
- 開発用リポジトリ：/home/lucifar/GitHub/ComfyUI-GGUF
  （コンテナの custom_nodes/ComfyUI-GGUF にバインドマウント済み。
    ファイルを編集すれば、ComfyUI再起動後にそのまま反映される）
- 再起動コマンド：systemctl --user restart comfyui-igpu
  （再起動は私が行う。Claude Codeは実行しない）
- custom_nodes のパス、コンテナ名、再起動方法：**不明な場合は推測せず、私に質問すること**

## 8. 禁止事項・注意

- 承認なしに `main` へ直接コミット・プッシュしない。作業ブランチで行う。
- 稼働中のComfyUIコンテナや `custom_nodes` を、確認なしに書き換えない・再起動しない。
- モデルファイルを削除・移動しない。
- ROCm専用のネイティブコード（HIPカーネル等）は今回の範囲外。PyTorchの範囲で対応する。
- 速度を得るために「ロード時に全重みを展開して保持する」方式を既定にしない
  （VRAM節約というGGUFの利点が消えるため）。採用する場合はオプション扱いにして事前に相談する。
- 原因が特定できない、または修正が大規模になりそうな場合は、作業を進めずに報告する。

## 9. 実施結果と未確認項目（2026-10-11 時点）

原因：同梱の K系 Triton 融合カーネル（`kquant_triton_backend.py`）が ROCm でも有効になり、
拡散モデルでは展開→rocBLAS より大幅に遅かった。ROCm では既定で無効化して対応
（`COMFYUI_GGUF_KQUANT_TRITON_ROCM=1` で再有効化可）。
切り分け用に `COMFYUI_GGUF_DEQUANT_PROFILE=1` の計測ログを追加（既定無効）。

### 確認済み

| 項目 | 結果 |
|---|---|
| Qwen-Image 2.1 Q6_K の速度 | 24.70 → **5.26 s/it**（目標 6.35 s/it 前後を達成） |
| Qwen-Image 2.1 Q8_0 の速度 | **4.52 s/it**（再起動直後に単独実行。劣化なし） |
| K系展開と `gguf.quants.dequantize` の数値一致 | Q2_K〜Q6_K で一致（`tests/test_kquant_dequant.py`、CPU / 8060S） |
| 既存テスト | 141件中、main 時点から失敗している 5件（Q4_CR / MiniMax H3 変換）以外は成功 |

### 未確認（後日、実機で確認する）

- [ ] Krea 2 の既存 GGUF が従来どおり読み込み・生成できること
- [ ] 修正前（または leejet 版）と同一シードの出力画像が一致（またはほぼ一致）すること
- [ ] LoRA 適用
- [ ] テキストエンコーダの GGUF 読み込み
- [ ] Q8_CR / Q4_CR
- [ ] メモリマップ読み込み（Dynamic VRAM 経路）

補足：Q6_K の直後に同じプロセスで Q8_0 へ切り替えると 5.91 s/it になった。
速度比較は ComfyUI 再起動直後に、対象モデルだけを実行して計測すること。

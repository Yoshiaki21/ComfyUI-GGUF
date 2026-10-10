"""K系展開経路の切り分け用プロファイラ（既定では無効）。

環境変数 ``COMFYUI_GGUF_DEQUANT_PROFILE=1`` で有効化すると、量子化重みの
展開（dequant）、展開後の matmul、K系ネイティブカーネルの所要時間を
量子化型・デバイスごとに集計し、一定間隔で ComfyUI のログへ出力する。
計測のため各区間の前後で GPU を同期するので、有効時は推論が遅くなる。
s/it の比較は必ず無効状態で行うこと。
"""

import logging
import os
import threading
import time
from contextlib import contextmanager

import torch

_ENV = "COMFYUI_GGUF_DEQUANT_PROFILE"
_INTERVAL_ENV = "COMFYUI_GGUF_DEQUANT_PROFILE_INTERVAL"

ENABLED = os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
try:
    _INTERVAL = max(1.0, float(os.environ.get(_INTERVAL_ENV, "30")))
except ValueError:
    _INTERVAL = 30.0

_lock = threading.Lock()
_stats = {}
_last_report = time.perf_counter()


def _sync(device):
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _qtype_name(qtype):
    return getattr(qtype, "name", None) or str(qtype)


@contextmanager
def timed(kind, qtype, device):
    """``kind`` 区間（dequant / matmul / native_kquant）の時間を計測する。

    yield する dict の ``record`` を False にすると、その呼び出しは集計しない
    （ネイティブ経路が辞退してフォールバックした場合など）。
    """
    state = {"record": True}
    _sync(device)
    started = time.perf_counter()
    try:
        yield state
    finally:
        _sync(device)
    # 例外時はここに来ないので、失敗した呼び出しは集計しない
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not state["record"]:
        return
    key = (kind, _qtype_name(qtype), getattr(device, "type", str(device)))
    with _lock:
        entry = _stats.setdefault(key, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += elapsed_ms
        entry[2] = max(entry[2], elapsed_ms)
    _maybe_report()


def _maybe_report(force=False):
    global _last_report
    now = time.perf_counter()
    with _lock:
        if not _stats or (not force and now - _last_report < _INTERVAL):
            return
        window = now - _last_report
        snapshot = sorted(_stats.items())
        _stats.clear()
        _last_report = now

    lines = [f"ComfyUI-GGUF dequant profile (last {window:.1f}s):"]
    for (kind, qtype, device), (count, total_ms, max_ms) in snapshot:
        lines.append(
            f"  {kind:<13} {qtype:<6} {device:<5} calls={count:<7d} "
            f"total={total_ms:10.1f}ms avg={total_ms / count:8.3f}ms max={max_ms:8.3f}ms"
        )
    logging.info("\n".join(lines))


def report_now():
    """溜まっている集計を即座に出力する（テスト・手動確認用）。"""
    _maybe_report(force=True)


if ENABLED:
    logging.info(
        "ComfyUI-GGUF: dequant profiling enabled (%s=1, interval %.0fs); "
        "timings synchronize the GPU and slow down inference",
        _ENV,
        _INTERVAL,
    )

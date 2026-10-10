"""K系量子化（Q2_K〜Q6_K）の PyTorch 展開を gguf の numpy 参照実装と比較するテスト。

CPU だけで実行できる。GPU（CUDA / ROCm）がある場合は、デバイス上の展開も追加で検証する。
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import gguf
import numpy as np
import torch

from dequant import dequantize, dequantize_tensor


Q = gguf.GGMLQuantizationType
ALL_K_QUANTS = (Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K)
# リポジトリ内の変換器にエンコーダがある型（gguf の Python 実装は K系の量子化が未実装）
ENCODABLE_K_QUANTS = (Q.Q4_K, Q.Q5_K, Q.Q6_K)

# ブロック内の fp16 スケール（d / dmin）のバイト位置。ggml-common.h の block_qX_K 定義に対応。
_FP16_OFFSETS = {
    Q.Q2_K: (80, 82),
    Q.Q3_K: (108,),
    Q.Q4_K: (0, 2),
    Q.Q5_K: (0, 2),
    Q.Q6_K: (208,),
}


def random_blocks(qtype, rows, cols, seed):
    """ランダムなビット列に、有限で小さい fp16 スケールを埋めた K系ブロックを作る。"""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    n_blocks = rows * cols // block_size
    blocks = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    for offset in _FP16_OFFSETS[qtype]:
        scale = rng.uniform(-0.05, 0.05, size=n_blocks).astype(np.float16)
        blocks[:, offset : offset + 2] = scale.view(np.uint8).reshape(n_blocks, 2)
    return blocks.reshape(rows, cols // block_size * type_size)


def reference_dequantize(raw, qtype, shape):
    return torch.from_numpy(
        gguf.quants.dequantize(np.ascontiguousarray(raw), qtype).astype(np.float32)
    ).reshape(shape)


def load_converter():
    try:
        import tools.convert as converter
    except Exception as error:  # ComfyUI がない環境では変換器を読み込めない
        raise unittest.SkipTest(f"tools.convert is unavailable: {error}")
    return converter


class KQuantDequantizeTests(unittest.TestCase):
    rows, cols = 8, 1024

    def assert_matches_reference(self, qtype, raw, device="cpu"):
        shape = (self.rows, self.cols)
        reference = reference_dequantize(raw, qtype, shape)
        data = torch.from_numpy(raw).to(device)

        # float32 で展開した場合は参照実装とほぼ完全に一致すること
        actual = dequantize(data, qtype, shape, dtype=torch.float32).cpu()
        max_error = (actual - reference).abs().max().item()
        self.assertLessEqual(max_error, 1e-6 * max(1.0, reference.abs().max().item()))

        # 実行時の既定経路（dequant_dtype=None → 計算 dtype へ変換）も許容誤差内であること
        tensor = data.clone()
        tensor.tensor_type = qtype
        tensor.tensor_shape = torch.Size(shape)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual = dequantize_tensor(tensor, dtype).float().cpu()
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=torch.finfo(dtype).eps * 4,
                    atol=reference.abs().max().item() * torch.finfo(dtype).eps * 4,
                )

    def test_random_blocks_match_gguf_reference(self):
        for qtype in ALL_K_QUANTS:
            with self.subTest(qtype=qtype.name):
                raw = random_blocks(qtype, self.rows, self.cols, seed=int(qtype))
                self.assert_matches_reference(qtype, raw)

    def test_quantize_roundtrip_matches_gguf_reference(self):
        converter = load_converter()
        rng = np.random.default_rng(1234)
        weight = rng.standard_normal((self.rows, self.cols)).astype(np.float32) * 0.02
        for qtype in ENCODABLE_K_QUANTS:
            with self.subTest(qtype=qtype.name):
                raw = np.ascontiguousarray(converter.quantize_k(weight, qtype))
                self.assert_matches_reference(qtype, raw)

                # 量子化誤差そのものも妥当な範囲であること（エンコーダとの整合性の確認）
                restored = reference_dequantize(raw, qtype, weight.shape).numpy()
                rel_rmse = np.sqrt(np.mean((restored - weight) ** 2)) / weight.std()
                self.assertLess(rel_rmse, 0.15)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU (CUDA/ROCm) is not available")
    def test_gpu_dequantize_matches_gguf_reference(self):
        for qtype in ALL_K_QUANTS:
            with self.subTest(qtype=qtype.name):
                raw = random_blocks(qtype, self.rows, self.cols, seed=100 + int(qtype))
                self.assert_matches_reference(qtype, raw, device="cuda")


def load_triton_backend():
    path = Path(__file__).parents[1] / "kquant_triton_backend.py"
    spec = importlib.util.spec_from_file_location("kquant_triton_backend_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TritonRocmGateTests(unittest.TestCase):
    """ROCm では同梱 Triton カーネルが既定で無効になり、展開経路へフォールバックすること。"""

    def setUp(self):
        self.module = load_triton_backend()
        # Triton が入っていない環境でも判定ロジックだけ検証できるようにする
        self.module.triton = object()

    def supports(self):
        return self.module._supports(
            qtype="Q6_K",
            device=torch.device("cuda", 0),
            input_dtype=torch.bfloat16,
            weight_shape=(3072, 3072),
            weight_dtype=torch.uint8,
        )

    def test_rocm_disabled_by_default(self):
        with mock.patch.object(torch.version, "hip", "7.0.0"), mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("COMFYUI_GGUF_KQUANT_TRITON_ROCM", None)
            with self.assertLogs(level="INFO"):
                self.assertFalse(self.supports())

    def test_rocm_opt_in(self):
        with mock.patch.object(torch.version, "hip", "7.0.0"), mock.patch.dict(
            os.environ, {"COMFYUI_GGUF_KQUANT_TRITON_ROCM": "1"}
        ):
            self.assertTrue(self.supports())

    def test_cuda_unchanged(self):
        with mock.patch.object(torch.version, "hip", None):
            self.assertTrue(self.supports())


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gguf
import torch


def load_runtime_modules():
    root = Path(__file__).parents[1]
    package_name = "comfyui_gguf_kquant_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package

    loaded = {}
    for module_name in ("quant_ops", "ops"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}", root / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    loaded["backend"] = sys.modules[f"{package_name}.kquant_backend"]
    loaded["profile"] = sys.modules[f"{package_name}.dequant_profile"]
    return loaded


class FakeKQuantBackend:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def supports(self, **kwargs):
        self.calls.append(("supports", kwargs))
        return True

    def linear(self, **kwargs):
        self.calls.append(("linear", kwargs))
        if self.fail:
            raise RuntimeError("synthetic kernel failure")
        weight = torch.ones(
            kwargs["weight_shape"],
            dtype=kwargs["input"].dtype,
            device=kwargs["input"].device,
        )
        return torch.nn.functional.linear(kwargs["input"], weight, kwargs["bias"])


class KQuantBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = load_runtime_modules()
        cls.ops = modules["ops"]
        cls.quant_ops = modules["quant_ops"]
        cls.backend_module = modules["backend"]
        cls.profile = modules["profile"]

    def tearDown(self):
        self.backend_module.register_kquant_backend(None)

    @staticmethod
    def quantized_storage(qtype, out_features=2):
        _, type_size = gguf.GGML_QUANT_SIZES[qtype]
        return torch.zeros(out_features * type_size, dtype=torch.uint8)

    def static_linear(self, qtype):
        raw = self.quantized_storage(qtype)
        weight = self.ops.GGMLTensor(
            raw,
            tensor_type=qtype,
            tensor_shape=(2, 256),
            compute_dtype=torch.float32,
        )
        linear = self.ops.GGMLOps.Linear(
            256, 2, bias=False, dtype=torch.float32
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        return linear, raw

    def test_static_linear_sends_all_supported_k_quants_directly_to_backend(self):
        for qtype in (
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            with self.subTest(qtype=qtype.name):
                backend = FakeKQuantBackend()
                self.backend_module.register_kquant_backend(backend)
                linear, raw = self.static_linear(qtype)
                input_tensor = torch.ones((2, 3, 256), dtype=torch.float32)

                output = linear.forward_ggml_cast_weights(input_tensor)

                self.assertEqual(tuple(output.shape), (2, 3, 2))
                self.assertTrue(torch.equal(output, torch.full_like(output, 256)))
                call = next(value for name, value in backend.calls if name == "linear")
                self.assertEqual(call["qtype"], qtype.name)
                self.assertEqual(call["weight_shape"], (2, 256))
                self.assertEqual(call["weight"].untyped_storage().nbytes(), raw.numel())

    def test_backend_failure_is_disabled_and_uses_dequantized_fallback(self):
        backend = FakeKQuantBackend(fail=True)
        self.backend_module.register_kquant_backend(backend)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q4_K)
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        first = linear.forward_ggml_cast_weights(input_tensor)
        second = linear.forward_ggml_cast_weights(input_tensor)

        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))
        self.assertEqual(
            sum(name == "linear" for name, _ in backend.calls),
            1,
        )

    def test_dynamic_layout_dispatches_linear_and_transposed_mm_without_dequant(self):
        backend = FakeKQuantBackend()
        self.backend_module.register_kquant_backend(backend)
        qtype = gguf.GGMLQuantizationType.Q5_K
        weight = self.quant_ops.make_quantized(
            self.quantized_storage(qtype),
            qtype,
            (2, 256),
            orig_dtype=torch.float32,
        )
        input_tensor = torch.ones((3, 256), dtype=torch.float32)
        bias = torch.tensor([1.0, 2.0])

        linear_output = torch.nn.functional.linear(input_tensor, weight, bias)
        mm_output = torch.mm(input_tensor, weight.t())

        self.assertTrue(
            torch.equal(
                linear_output,
                torch.tensor([[257.0, 258.0]]).expand(3, 2),
            )
        )
        self.assertTrue(torch.equal(mm_output, torch.full((3, 2), 256.0)))
        self.assertEqual(
            sum(name == "linear" for name, _ in backend.calls),
            2,
        )

    def test_missing_optional_backend_uses_cpu_dequantized_fallback(self):
        self.backend_module.register_kquant_backend(None)
        error = ModuleNotFoundError("No optional test backend")
        error.name = "comfyui_gguf_kquant"
        qtype = gguf.GGMLQuantizationType.Q6_K
        weight = self.quant_ops.make_quantized(
            self.quantized_storage(qtype),
            qtype,
            (2, 256),
            orig_dtype=torch.float32,
        )
        input_tensor = torch.ones((1, 256), dtype=torch.float32)

        with mock.patch.object(
            self.backend_module.importlib,
            "import_module",
            side_effect=error,
        ):
            output = torch.nn.functional.linear(input_tensor, weight)

        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_invalid_k_quant_storage_never_reaches_backend(self):
        backend = FakeKQuantBackend()
        self.backend_module.register_kquant_backend(backend)
        raw = torch.zeros((2, 143), dtype=torch.uint8)

        output = self.backend_module.try_kquant_linear(
            torch.ones((1, 256)),
            raw,
            gguf.GGMLQuantizationType.Q4_K,
            (2, 256),
        )

        self.assertIsNone(output)
        self.assertEqual(backend.calls, [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the bundled Triton backend")
    # ROCm では同梱カーネルが既定で無効なため、数値検証のときだけ明示的に有効化する。
    @mock.patch.dict("os.environ", {"COMFYUI_GGUF_KQUANT_TRITON_ROCM": "1"})
    def test_bundled_triton_backend_matches_reference(self):
        self.backend_module.register_kquant_backend(None)
        from dequant import dequantize

        for qtype in (
            gguf.GGMLQuantizationType.Q4_K,
            gguf.GGMLQuantizationType.Q5_K,
            gguf.GGMLQuantizationType.Q6_K,
        ):
            with self.subTest(qtype=qtype.name):
                _, type_size = gguf.GGML_QUANT_SIZES[qtype]
                raw = torch.randint(
                    0, 256, (3, 2 * type_size), device="cuda", dtype=torch.uint8
                )
                for offset in (0, type_size):
                    if qtype == gguf.GGMLQuantizationType.Q6_K:
                        raw[:, offset + 192 : offset + 208] = torch.randint(
                            0, 10, (3, 16), device="cuda", dtype=torch.uint8
                        )
                        raw[:, offset + 208] = 0
                        raw[:, offset + 209] = 60
                    else:
                        raw[:, offset] = 0
                        raw[:, offset + 1] = 60
                        raw[:, offset + 2 : offset + 4] = 0
                        raw[:, offset + 4 : offset + 16] = torch.randint(
                            0, 64, (3, 12), device="cuda", dtype=torch.uint8
                        )
                input_tensor = torch.randn(
                    2, 5, 512, device="cuda", dtype=torch.float16
                )
                bias = torch.randn(3, device="cuda", dtype=input_tensor.dtype)
                output = self.backend_module.try_kquant_linear(
                    input_tensor, raw, qtype, (3, 512), bias=bias
                )
                reference_weight = dequantize(
                    raw.cpu(), qtype, (3, 512), dtype=input_tensor.dtype
                ).cuda()
                reference = torch.nn.functional.linear(
                    input_tensor, reference_weight, bias
                )
                torch.testing.assert_close(output, reference, rtol=2e-3, atol=8)


class DequantProfileTests(unittest.TestCase):
    """COMFYUI_GGUF_DEQUANT_PROFILE の切り分け用計測が経路ごとに集計されること。"""

    @classmethod
    def setUpClass(cls):
        modules = load_runtime_modules()
        cls.ops = modules["ops"]
        cls.backend_module = modules["backend"]
        cls.profile = modules["profile"]

    quantized_storage = staticmethod(KQuantBackendTests.quantized_storage)
    static_linear = KQuantBackendTests.static_linear

    def setUp(self):
        self.profile._stats.clear()
        patcher = mock.patch.object(self.profile, "ENABLED", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.profile._stats.clear)
        self.addCleanup(self.backend_module.register_kquant_backend, None)

    def recorded(self):
        return {key: value[0] for key, value in self.profile._stats.items()}

    def test_native_route_is_recorded(self):
        self.backend_module.register_kquant_backend(FakeKQuantBackend())
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q4_K)
        linear.forward_ggml_cast_weights(torch.ones((1, 256), dtype=torch.float32))
        self.assertEqual(self.recorded(), {("native_kquant", "Q4_K", "cpu"): 1})

    def test_declined_native_route_records_dequant_and_matmul(self):
        backend = FakeKQuantBackend()
        backend.supports = lambda **kwargs: False
        self.backend_module.register_kquant_backend(backend)
        linear, _ = self.static_linear(gguf.GGMLQuantizationType.Q6_K)
        linear.forward_ggml_cast_weights(torch.ones((1, 256), dtype=torch.float32))
        self.assertEqual(
            self.recorded(),
            {("dequant", "Q6_K", "cpu"): 1, ("matmul", "Q6_K", "cpu"): 1},
        )
        with self.assertLogs(level="INFO") as logs:
            self.profile.report_now()
        self.assertIn("dequant", logs.output[0])
        self.assertEqual(self.profile._stats, {})

    @unittest.skipUnless(
        torch.cuda.is_available() and getattr(torch.version, "hip", None),
        "ROCm GPU is required",
    )
    def test_rocm_q6k_linear_uses_dequantized_route(self):
        # 実機（ROCm）で同梱 Triton カーネルを経由せず、展開→matmul 経路になること
        from tests.test_kquant_dequant import random_blocks

        self.backend_module.register_kquant_backend(None)
        self.backend_module._backend_checked = False
        qtype = gguf.GGMLQuantizationType.Q6_K
        raw = torch.from_numpy(random_blocks(qtype, 64, 512, seed=7)).cuda()
        weight = self.ops.GGMLTensor(
            raw, tensor_type=qtype, tensor_shape=torch.Size((64, 512))
        )
        linear = self.ops.GGMLOps.Linear(512, 64, bias=False, dtype=torch.bfloat16)
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        input_tensor = torch.randn((2, 300, 512), device="cuda", dtype=torch.bfloat16)

        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("COMFYUI_GGUF_KQUANT_TRITON_ROCM", None)
            output = linear.forward_ggml_cast_weights(input_tensor)

        from dequant import dequantize

        # 精度そのものは test_kquant_dequant で検証済み。ここでは経路の確認のため bf16 の丸め幅で比較する
        reference = torch.nn.functional.linear(
            input_tensor.float(), dequantize(raw, qtype, (64, 512), dtype=torch.float32)
        )
        scale = reference.abs().max().item()
        torch.testing.assert_close(output.float(), reference, rtol=0, atol=scale * 2e-2)
        recorded = self.recorded()
        self.assertNotIn(("native_kquant", "Q6_K", "cuda"), recorded)
        self.assertEqual(recorded.get(("dequant", "Q6_K", "cuda")), 1)
        self.assertEqual(recorded.get(("matmul", "Q6_K", "cuda")), 1)


if __name__ == "__main__":
    unittest.main()

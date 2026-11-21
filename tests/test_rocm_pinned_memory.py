"""ROCm でページ境界に揃っていない重みをピン留めしないこと（GPU メモリアクセス違反の回避）。"""
import unittest
from unittest import mock

import torch

from tests.test_targeted_quantization import nodes_factory


class RocmUnalignedPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = nodes_factory()

    def model_with(self, tensor):
        model = torch.nn.Module()
        model.layer = torch.nn.Linear(1, 1, bias=False)
        model.layer.weight = torch.nn.Parameter(tensor, requires_grad=False)
        return model

    def unaligned(self):
        # 先頭 1 要素ずらしたビューは必ずページ境界からずれる
        base = torch.zeros(self.nodes._HOST_PAGE_SIZE * 2, dtype=torch.uint8)
        view = base[1:]
        self.assertNotEqual(view.data_ptr() % self.nodes._HOST_PAGE_SIZE, 0)
        return view

    def test_skips_unaligned_weight_on_rocm(self):
        model = self.model_with(self.unaligned())
        with mock.patch.object(torch.version, "hip", "7.0"):
            self.assertTrue(self.nodes._rocm_unaligned_pin(model, "layer.weight"))

    def test_keeps_pinning_on_cuda(self):
        model = self.model_with(self.unaligned())
        with mock.patch.object(torch.version, "hip", None):
            self.assertFalse(self.nodes._rocm_unaligned_pin(model, "layer.weight"))

    def test_keeps_pinning_page_aligned_weight_on_rocm(self):
        page = self.nodes._HOST_PAGE_SIZE
        base = torch.zeros(page * 3, dtype=torch.uint8)
        offset = (-base.data_ptr()) % page
        model = self.model_with(base[offset:offset + page])
        with mock.patch.object(torch.version, "hip", "7.0"):
            self.assertFalse(self.nodes._rocm_unaligned_pin(model, "layer.weight"))


if __name__ == "__main__":
    unittest.main()

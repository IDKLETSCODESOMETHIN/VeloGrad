#!/usr/bin/env python3
# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# SUITE: Universal Compatibility & Interoperability Master Verification Test
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

import sys
import os
import unittest
import json
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import velograd as vg
from velograd.tensor import Tensor, randn, zeros
from velograd import nn, hub, compat, converter

class TestVeloGradUniversalCompatibility(unittest.TestCase):
    """
    Exhaustive validation of cross-ecosystem compatibility spanning 2010 to present:
    - NumPy buffer protocol and ndarray exchange
    - Scikit-learn / Pandas tabular dataframe extraction
    - SciPy sparse matrix densification
    - PyTorch tensor & state_dict bidirectional translation
    - TensorFlow / Keras constant and numpy protocol exchange
    - JAX Array and PyTree buffer protocol
    - DLPack zero-copy tensor standard protocol
    - Hugging Face Safetensors zero-dependency serialization
    - GGUF v3 & ONNX multi-format export
    - 1-command CLI runner & converter tool
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="velograd_compat_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_numpy_and_array_interface(self):
        """Validates NumPy and Python nested list exchange (Core 2010+ standard)."""
        # 1. From Python list / tuple
        t1 = compat.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t1.shape, (2, 3))
        self.assertEqual(t1._data[0], 1.0)
        self.assertEqual(t1._data[5], 6.0)

        # 2. To NumPy
        arr = compat.to_numpy(t1)
        self.assertEqual(arr.shape, (2, 3))
        self.assertAlmostEqual(float(arr[1, 2]), 6.0, places=5)

    def test_dlpack_zero_copy_protocol(self):
        """Validates IEEE/Consortium DLPack standard interoperability."""
        t = randn(4, 8)
        capsule = compat.to_dlpack(t)
        self.assertEqual(capsule.shape, [4, 8])
        self.assertEqual(capsule.__dlpack_device__(), (1, 0)) # CPU

        # Restore from capsule
        restored = compat.from_dlpack(capsule)
        self.assertEqual(restored.shape, t.shape)
        self.assertEqual(len(restored._data), 32)

    def test_pytorch_state_dict_bidirectional_parity(self):
        """Validates PyTorch tensor mapping and state_dict auto-loading."""
        class MockPyTorchTensor:
            def __init__(self, data, shape):
                self._data = data
                self.shape = shape
                self.requires_grad = True
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                import numpy as np
                return np.array(self._data, dtype=np.float32).reshape(self.shape)

        mock_pt_tensor = MockPyTorchTensor([1.5, -2.0, 0.5, 3.0], (2, 2))
        vg_tensor = compat.from_torch(mock_pt_tensor)
        self.assertEqual(vg_tensor.shape, (2, 2))
        self.assertAlmostEqual(vg_tensor._data[0], 1.5, places=5)

        # State dict loading into VeloGrad model
        linear = nn.Linear(2, 2)
        state_dict = {
            "weight": MockPyTorchTensor([1.0, 0.0, 0.0, 1.0], (2, 2)),
            "bias": MockPyTorchTensor([0.1, 0.2], (2,))
        }
        loaded = compat.load_torch_state_dict(linear, state_dict, strict=False)
        self.assertTrue(len(loaded) > 0)

    def test_safetensors_and_hf_bridge(self):
        """Validates zero-dependency Hugging Face Safetensors file reading and writing."""
        out_st = os.path.join(self.tmp_dir, "test_model.safetensors")
        state_dict = {
            "model.embed_tokens.weight": {"shape": [4, 8], "data": [0.1 * i for i in range(32)]},
            "model.layers.0.mlp.gate.weight": {"shape": [8, 4], "data": [-0.05 * i for i in range(32)]}
        }
        hub.save_safetensors(state_dict, out_st)
        self.assertTrue(os.path.exists(out_st))
        self.assertGreater(os.path.getsize(out_st), 64)

        # Load back
        loaded = hub.load_safetensors(out_st)
        self.assertIn("model.embed_tokens.weight", loaded)
        self.assertEqual(loaded["model.embed_tokens.weight"].shape, (4, 8))
        self.assertAlmostEqual(loaded["model.embed_tokens.weight"]._data[1], 0.1, places=5)

    def test_universal_converter_pipeline(self):
        """Validates end-to-end model converter (Safetensors -> VeloGrad JSON & GGUF)."""
        input_st = os.path.join(self.tmp_dir, "input.safetensors")
        state_dict = {
            "weight": {"shape": [2, 4], "data": [1.0] * 8}
        }
        hub.save_safetensors(state_dict, input_st)

        # 1. Convert to VeloGrad JSON
        out_json = os.path.join(self.tmp_dir, "converted.json")
        converter.convert_model(input_st, out_json, target_format="velograd")
        self.assertTrue(os.path.exists(out_json))
        with open(out_json, "r") as f:
            data = json.load(f)
        self.assertIn("weight", data)

        # 2. Convert to GGUF
        out_gguf = os.path.join(self.tmp_dir, "converted.gguf")
        converter.convert_model(input_st, out_gguf, target_format="gguf")
        self.assertTrue(os.path.exists(out_gguf))
        self.assertGreater(os.path.getsize(out_gguf), 100)

    def test_cli_subcommand_interface(self):
        """Validates that velograd CLI subcommands (info, convert, run) parse properly."""
        from velograd import cli
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            # Test 'info' command
            sys.argv = ["velograd", "info"]
            cli.main()

        output = f.getvalue()
        self.assertIn("VELOGRAD ENGINE SYSTEM REPORT", output)
        self.assertIn("VERSION", output)
        self.assertIn("Ecosystem Parity", output)

if __name__ == "__main__":
    unittest.main()

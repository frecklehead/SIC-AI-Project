import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bsrnn_tse import build_model
from model_adapter import Runner


CAUSAL_V2_CFG = {
    "n_fft": 512,
    "hop": 256,
    "emb_dim": 192,
    "spk_hidden": 256,
    "causal": True,
    "band_bidir": True,
    "N": 96,
    "hidden": 192,
    "layers": 5,
}


class ModelConfigTests(unittest.TestCase):
    def test_causal_v2_checkpoint_loads_and_runs(self):
        source_model = build_model(CAUSAL_V2_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "causal_v2.pt"
            torch.save({
                "cfg": {"model": CAUSAL_V2_CFG},
                "model": source_model.state_dict(),
            }, checkpoint)

            runner = Runner(str(checkpoint), device="cpu")

        self.assertTrue(runner.causal)
        self.assertEqual(
            tuple(runner.model.state_dict()["layers.0.time.proj.weight"].shape),
            (96, 192),
        )
        self.assertNotIn(
            "layers.0.time.rnn.weight_ih_l0_reverse",
            runner.model.state_dict(),
        )
        self.assertIn(
            "layers.0.band.rnn.weight_ih_l0_reverse",
            runner.model.state_dict(),
        )

        runner.cond_t = torch.randn(1, CAUSAL_V2_CFG["emb_dim"])
        result = runner(np.zeros(2048, dtype=np.float32))
        self.assertEqual(result.shape, (2048,))
        self.assertTrue(np.isfinite(result).all())

    def test_default_architecture_remains_noncausal(self):
        model = build_model()
        state = model.state_dict()

        self.assertEqual(tuple(state["layers.0.time.proj.weight"].shape), (128, 512))
        self.assertIn("layers.0.time.rnn.weight_ih_l0_reverse", state)
        self.assertFalse(model.causal)


if __name__ == "__main__":
    unittest.main()

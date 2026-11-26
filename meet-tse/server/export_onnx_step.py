"""Export and smoke-test one causal BSRNN chunk with explicit LSTM states."""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model_adapter import Runner
from onnx_step import export_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", default="stateful_step.onnx")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    runner = Runner(args.ckpt, device="cpu")
    if not runner.causal:
        raise RuntimeError("checkpoint is not causal")
    runner.cond_t = torch.zeros(1, 192)
    wrapper, spec, states = export_step(runner, args.output)
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    spk = runner.model.spk_enc(torch.nn.functional.normalize(runner.cond_t.float(), dim=-1))
    inputs = {
        "spec_ri": np.random.default_rng(7).standard_normal(spec.shape).astype(np.float32),
        "spk": spk.detach().numpy(),
    }
    names = [i.name for i in sess.get_inputs()]
    for name, state in zip(names[2:], states):
        inputs[name] = state.numpy()
    with torch.inference_mode():
        eager = wrapper(torch.from_numpy(inputs["spec_ri"]), torch.from_numpy(inputs["spk"]), *states)
    actual = sess.run(None, inputs)
    errs = [float(np.max(np.abs(e.detach().numpy() - a))) for e, a in zip(eager, actual)]
    print(f"exported={args.output} ONNX nodes={len(model.graph.node)} inputs={len(inputs)}")
    print(f"max_abs_error mask/states={max(errs):.8g}")


if __name__ == "__main__":
    main()

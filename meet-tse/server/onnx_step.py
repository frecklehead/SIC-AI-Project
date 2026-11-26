"""ONNX wrapper for one causal model chunk with explicit LSTM states."""
import torch


class CausalStep(torch.nn.Module):
    def __init__(self, runner):
        super().__init__()
        self.model = runner.model
        self.ops = runner.batched_ops

    def forward(self, spec_ri, spk, *states):
        spec_features = self.ops.band_split(spec_ri)
        hidden = spec_features + self.model.spk_add(spk)[:, None, None]
        next_states = []
        for i, layer in enumerate(self.model.layers):
            b, k, t, n = hidden.shape
            g, beta = layer.film(spk).chunk(2, -1)
            x = hidden * (1 + g[:, None, None]) + beta[:, None, None]
            xr = x.reshape(b * k, t, n)
            h0, c0 = states[2 * i], states[2 * i + 1]
            y, (h1, c1) = layer.time.rnn(layer.time.norm(xr), (h0, c0))
            xr = xr + layer.time.proj(y)
            x = xr.reshape(b, k, t, n).permute(0, 2, 1, 3).reshape(b * t, k, n)
            x = layer.band(x).reshape(b, t, k, n).permute(0, 2, 1, 3)
            hidden = x
            next_states.extend((h1, c1))
        mask_ri = self.ops.decoder(hidden)
        return (mask_ri, *next_states)


def zero_states(runner, device="cpu"):
    states = []
    for layer in runner.model.layers:
        shape = (1, len(runner.model.bands), layer.time.rnn.hidden_size)
        states.extend((torch.zeros(shape, device=device), torch.zeros(shape, device=device)))
    return tuple(states)


def export_step(runner, path):
    wrapper = CausalStep(runner).eval()
    spec = torch.zeros(1, runner.model.n_fft // 2 + 1, 4, 2)
    spk = runner.model.spk_enc(torch.nn.functional.normalize(runner.cond_t.float(), dim=-1))
    states = zero_states(runner)
    input_names = ["spec_ri", "spk"]
    for i in range(len(runner.model.layers)):
        input_names.extend((f"h{i}", f"c{i}"))
    output_names = ["mask_ri"]
    for i in range(len(runner.model.layers)):
        output_names.extend((f"hout{i}", f"cout{i}"))
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (spec, spk, *states), path, input_names=input_names,
            output_names=output_names, opset_version=18, do_constant_folding=True,
            dynamo=False,
        )
    return wrapper, spec, states

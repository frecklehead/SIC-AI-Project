"""Runtime grouped band operations; checkpoint modules remain the source weights."""
import torch


class BatchedBandOps:
    """Stack per-band affine parameters and evaluate bands in width groups."""

    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.width_groups = {}
        for index, (start, end) in enumerate(model.bands):
            self.width_groups.setdefault(end - start, []).append(index)
        self.split_groups = []
        self.decoder_groups = []
        split = model.band_split
        dec = model.decoder
        for width, indices in self.width_groups.items():
            ids = torch.tensor(indices, dtype=torch.long, device=self.device)
            self.split_groups.append({
                "width": width,
                "indices": ids,
                "starts": [model.bands[i][0] for i in indices],
                "ln_w": torch.stack([split.norms[i].weight.detach() for i in indices]),
                "ln_b": torch.stack([split.norms[i].bias.detach() for i in indices]),
                "fc_w": torch.stack([split.fcs[i].weight.detach() for i in indices]),
                "fc_b": torch.stack([split.fcs[i].bias.detach() for i in indices]),
                "eps": split.norms[indices[0]].eps,
            })
            mlps = [dec.mlps[i] for i in indices]
            self.decoder_groups.append({
                "width": width,
                "indices": ids,
                "ln_w": torch.stack([dec.norms[i].weight.detach() for i in indices]),
                "ln_b": torch.stack([dec.norms[i].bias.detach() for i in indices]),
                "fc1_w": torch.stack([mlp[0].weight.detach() for mlp in mlps]),
                "fc1_b": torch.stack([mlp[0].bias.detach() for mlp in mlps]),
                "fc2_w": torch.stack([mlp[2].weight.detach() for mlp in mlps]),
                "fc2_b": torch.stack([mlp[2].bias.detach() for mlp in mlps]),
                "eps": dec.norms[indices[0]].eps,
            })

    @staticmethod
    def _norm(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + eps) * weight[None, :, None, :] + bias[None, :, None, :]

    def band_split(self, spec):
        # x: [B,F,T,2] -> per-band [B,K,T,2*width]
        x = torch.view_as_real(spec) if torch.is_complex(spec) else spec
        b, _, t, _ = x.shape
        outputs = [None] * sum(len(g["starts"]) for g in self.split_groups)
        for group in self.split_groups:
            width, indices = group["width"], group["indices"]
            bands = torch.stack([x[:, s:s + width] for s in group["starts"]], dim=1)
            bands = bands.permute(0, 1, 3, 2, 4).reshape(b, len(indices), t, 2 * width)
            z = self._norm(bands, group["ln_w"], group["ln_b"], group["eps"])
            z = torch.einsum("bktd,knd->bktn", z, group["fc_w"]) + group["fc_b"][None, :, None, :]
            for j, idx in enumerate(indices.tolist()):
                outputs[idx] = z[:, j]
        return torch.stack(outputs, dim=1)

    def decoder(self, hidden):
        # hidden: [B,K,T,N] -> concatenated [B,F,T,2] real/imag mask
        outputs = [None] * sum(len(g["indices"]) for g in self.decoder_groups)
        for group in self.decoder_groups:
            indices = group["indices"]
            x = hidden.index_select(1, indices)
            x = self._norm(x, group["ln_w"], group["ln_b"], group["eps"])
            x = torch.tanh(torch.einsum("bktn,kon->bkto", x, group["fc1_w"]) + group["fc1_b"][None, :, None, :])
            x = torch.einsum("bkto,kpo->bktp", x, group["fc2_w"]) + group["fc2_b"][None, :, None, :]
            # Match nn.GLU(dim=-1): split the final dimension and gate.
            left, gate = x.chunk(2, dim=-1)
            x = left * torch.sigmoid(gate)
            width = group["width"]
            x = x.reshape(x.shape[0], len(indices), x.shape[2], width, 2).permute(0, 1, 3, 2, 4)
            for j, idx in enumerate(indices.tolist()):
                outputs[idx] = x[:, j]
        return torch.cat(outputs, dim=1)

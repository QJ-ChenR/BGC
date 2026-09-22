"""A versioned ESMC adapter with bounded batches and explicit residue pooling."""

from importlib.metadata import version
from pathlib import Path

import torch

from .common import MAX_RESIDUES, file_digest


MODEL_SPECS = {"esmc_300m": (960, 15, 30, "300m"),
               "esmc_600m": (1152, 18, 36, "600m")}
ESM_VERSION = "3.2.1"


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    return device


class ESMCEncoder:
    def __init__(self, model_name="esmc_600m", device="cuda", weights=None,
                 batch_size=2, token_budget=4096):
        if version("esm") != ESM_VERSION:
            raise ValueError(f"This adapter requires esm=={ESM_VERSION}; install requirements-server.txt")
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
        from huggingface_hub import hf_hub_download

        if model_name not in MODEL_SPECS or batch_size < 1 or token_budget < 2048:
            raise ValueError("Invalid ESMC model, batch size, or token budget (minimum 2048)")
        self.device = select_device(device)
        self.batch_size, self.token_budget = batch_size, token_budget
        dimension, heads, layers, size = MODEL_SPECS[model_name]
        if weights is None:
            weights = hf_hub_download(
                repo_id=f"EvolutionaryScale/esmc-{size}-2024-12",
                filename=f"data/weights/esmc_{size}_2024_12_v0.pth")
        weights = Path(weights)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA workflow requires BF16 support")
        self.identity = {"model": model_name, "embedding_dim": dimension,
                         "esm_version": ESM_VERSION, "weights_sha256": file_digest(weights),
                         "precision": str(dtype), "pooling": "last_output_residue_mean_v1",
                         "attention": "pytorch_sdpa", "max_residues": MAX_RESIDUES,
                         "torch_version": str(torch.__version__)}
        # Use PyTorch SDPA; no separately compiled flash-attn extension is required.
        self.model = ESMC(d_model=dimension, n_heads=heads, n_layers=layers,
                          tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def pool(self, requests):
        """Pool [start, end) of each request sequence, excluding BOS/EOS/padding."""
        for sequence, start, end in requests:
            if not 0 <= start < end <= len(sequence) <= MAX_RESIDUES:
                raise ValueError("Invalid residue pooling interval or ESMC sequence length")
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][0]))
        results, cursor = [None] * len(requests), 0
        while cursor < len(order):
            selected = []
            while cursor < len(order) and len(selected) < self.batch_size:
                candidate = order[cursor]
                padded_tokens = (len(requests[candidate][0]) + 2) * (len(selected) + 1)
                if selected and padded_tokens > self.token_budget:
                    break
                selected.append(candidate)
                cursor += 1
            tokens = []
            for index in selected:
                sequence = requests[index][0]
                encoded = self.model.tokenizer(sequence, add_special_tokens=True)["input_ids"]
                if len(encoded) != len(sequence) + 2:
                    raise ValueError("Tokenizer did not produce exactly one token per residue plus BOS/EOS")
                tokens.append(torch.tensor(encoded, dtype=torch.long))
            padded = torch.nn.utils.rnn.pad_sequence(
                tokens, batch_first=True, padding_value=self.model.tokenizer.pad_token_id).to(self.device)
            output = self.model(sequence_tokens=padded)
            for row, index in enumerate(selected):
                _, start, end = requests[index]
                vector = output.embeddings[row, start + 1:end + 1].float().mean(dim=0).cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError("ESMC produced non-finite embeddings")
                results[index] = vector
            del output
        return results

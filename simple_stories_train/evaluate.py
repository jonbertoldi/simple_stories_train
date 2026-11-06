"""Evaluate the final (or a given) checkpoint of one or more runs on a whole dataset split.

The val loss logged during training averages only `val_max_steps` batches, whose count depends on
`batch_size`. This script averages the loss over every full batch of the split, so runs trained
with different settings are compared on exactly the same tokens.

Usage:
    python simple_stories_train/evaluate.py out/run1_5M/<timestamp> out/run2_11M/<timestamp>

An argument that is not a local directory is loaded from the HuggingFace Hub instead, e.g.
SimpleStories/SimpleStories-30M, to compare against the published models.
"""

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import fire
import torch
import yaml
from torch.utils.data import DataLoader

from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS

PACKAGE_DIR = Path(__file__).parent


def load_run(run_dir: Path, step: int | None, device: str) -> tuple[Llama, dict[str, Any], int]:
    with open(run_dir / "checkpoints" / "final_config.yaml") as f:
        config = yaml.safe_load(f)
    checkpoints = {
        int(p.stem.removeprefix("model_step_")): p
        for p in (run_dir / "checkpoints").glob("model_step_*.pt")
    }
    step = max(checkpoints) if step is None else step
    state_dict = torch.load(checkpoints[step], map_location=device, weights_only=True)
    model = Llama(MODEL_CONFIGS[config["model_name"]])
    # Runs compiled before the fix in train_llama.py saved every key with a "_orig_mod." prefix
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state_dict.items()})
    return model.to(device).eval(), config, step


def load_hub_model(model_id: str, device: str) -> tuple[Llama, dict[str, Any]]:
    """Load a published model such as SimpleStories/SimpleStories-30M (size from the id suffix).

    It is evaluated on the val split of the repo's own config, which uses the same dataset and
    tokenizer as the published SimpleStories models.
    """
    model = Llama.from_pretrained(model_id, MODEL_CONFIGS[model_id.rsplit("-", 1)[1]])
    with open(PACKAGE_DIR / "35M_config.yaml") as f:
        config = yaml.safe_load(f)
    return model.to(device).eval(), config


@torch.no_grad()
def mean_loss(model: Llama, loader: DataLoader[Any], device: str) -> tuple[float, int]:
    # Same precision as the val loss during training: bf16 autocast on CUDA only
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in device else nullcontext()
    total, num_batches = 0.0, 0
    for batch in loader:
        bat = batch["input_ids"].to(device, torch.long)
        with ctx:
            _, loss = model(bat[:, :-1], bat[:, 1:].contiguous(), return_logits=False)
        total += loss.item()
        num_batches += 1
    # Every batch has the same number of tokens (drop_last), so this is the mean over tokens
    return total / num_batches, num_batches


def main(
    *run_dirs: str,
    step: int | None = None,
    split: str = "test",
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print(f"{'run':<60} {'step':>6} {'loss':>7} {'ppl':>6}")
    used_step: int | str
    for run_dir in map(Path, run_dirs):
        if run_dir.exists():
            model, config, used_step = load_run(run_dir, step, device)
        else:
            model, config = load_hub_model(str(run_dir), device)
            used_step = "hub"
        data_config = DatasetConfig(**config["val_dataset_config"])
        data_config = data_config.model_copy(
            update={
                "split": split,
                # The path in the config is relative to the package dir, where training runs
                "tokenizer_file_path": str(PACKAGE_DIR / str(data_config.tokenizer_file_path)),
            }
        )
        loader, _ = create_data_loader(data_config, batch_size=batch_size)
        loss, num_batches = mean_loss(model, loader, device)
        print(
            f"{str(run_dir):<60} {used_step:>6} {loss:>7.4f} {math.exp(loss):>6.2f} "
            f"({num_batches} batches of {batch_size})",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)

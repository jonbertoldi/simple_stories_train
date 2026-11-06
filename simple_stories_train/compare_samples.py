"""Generate text from the snapshots of several training runs, side by side.

Every run gets the same seeds at every step, so the differences between columns come from the
weights and not from the sampling luck.

Usage:
    python simple_stories_train/compare_samples.py out/run1_5M/<timestamp> out/run2_11M/<timestamp>
"""

from pathlib import Path

import fire
import torch
import yaml
from tokenizers import Tokenizer

from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS

PACKAGE_DIR = Path(__file__).parent


def load_run(run_dir: Path) -> tuple[str, Tokenizer]:
    with open(run_dir / "checkpoints" / "final_config.yaml") as f:
        config = yaml.safe_load(f)
    # The tokenizer path in the config is relative to the package dir, where training is launched
    tokenizer_path = PACKAGE_DIR / config["train_dataset_config"]["tokenizer_file_path"]
    return config["model_name"], Tokenizer.from_file(str(tokenizer_path))


def main(
    *run_dirs: str,
    steps: tuple[int, ...] = (64, 256, 1000, 2000, 4000, 9155),
    num_samples: int = 3,
    max_new_tokens: int = 48,
    temperature: float = 1.0,
    top_k: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    runs = [(Path(d), *load_run(Path(d))) for d in run_dirs]
    eos_id = runs[0][2].token_to_id("[EOS]")

    print(
        f"Sampling: temperature {temperature}, top_k {top_k}, {max_new_tokens} new tokens, "
        f"each sample starts from [EOS], sample i uses seed {seed} + i in every run.\n"
    )
    for step in steps:
        print(f"## Step {step}\n")
        for run_dir, model_name, tokenizer in runs:
            model = Llama(MODEL_CONFIGS[model_name])
            state_dict = torch.load(
                run_dir / "checkpoints" / f"model_step_{step}.pt",
                map_location=device,
                weights_only=True,
            )
            # Runs compiled before the fix in train_llama.py saved keys with a "_orig_mod." prefix
            model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state_dict.items()})
            model.to(device).eval()
            print(f"**{model_name}**\n")
            for i in range(num_samples):
                torch.manual_seed(seed + i)
                idx = torch.tensor([[eos_id]], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
                print(f"{i + 1}. {tokenizer.decode(out[0, 1:].tolist())}")
            print()


if __name__ == "__main__":
    fire.Fire(main)

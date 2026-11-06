"""Generate stories from one or more models, side by side.

Each argument is a run directory (its final checkpoint, or --step) or a HuggingFace Hub id such as
SimpleStories/SimpleStories-30M. Sample i uses seed + i for every model, so the differences between
models come from the weights and not from the sampling luck. Generation stops at [EOS].

Usage:
    python simple_stories_train/generate.py out/final_30M_600M/<timestamp> \
        SimpleStories/SimpleStories-30M --prompt "the curious cat" --temperature 0.8
"""

from pathlib import Path

import fire
import torch
from tokenizers import Tokenizer

from simple_stories_train.evaluate import PACKAGE_DIR, load_hub_model, load_run


def main(
    *models: str,
    prompt: str = "",
    num_samples: int = 3,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int | None = None,
    step: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    print(
        f"Sampling: temperature {temperature}, top_k {top_k}, up to {max_new_tokens} new tokens, "
        f"prompt {prompt!r}, sample i uses seed {seed} + i for every model.\n"
    )
    for name in models:
        if Path(name).exists():
            model, config, used_step = load_run(Path(name), step, device)
            label = f"{name} (step {used_step})"
        else:
            model, config = load_hub_model(name, device)
            label = name
        tokenizer_path = PACKAGE_DIR / config["val_dataset_config"]["tokenizer_file_path"]
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        eos_id = tokenizer.token_to_id("[EOS]")
        # Start from [EOS], like every story in training, then the prompt (lowercased by the
        # tokenizer's normalizer, as in training)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        idx = torch.tensor([[eos_id, *prompt_ids]], dtype=torch.long, device=device)

        print(f"## {label}\n")
        for i in range(num_samples):
            torch.manual_seed(seed + i)
            with torch.no_grad():
                out = model.generate(
                    idx, max_new_tokens, temperature=temperature, top_k=top_k, eos_token_id=eos_id
                )
            print(f"{i + 1}. {tokenizer.decode(out[0, 1:].tolist())}\n")


if __name__ == "__main__":
    fire.Fire(main)

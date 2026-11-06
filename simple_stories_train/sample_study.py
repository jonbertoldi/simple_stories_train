"""Measure generated text over many samples, to compare sampling settings, training steps or models.

Metrics, on the generated tokens only (prompt excluded):
- invented: share of generated words never seen in the first 200k training stories (noise; rare
  real words count too)
- loops: share of samples where some 4-token sequence appears 3+ times (repetition)
- ended: share of samples that reach [EOS] within the token budget
- keeps (prompts mode): share of samples whose second half still names the prompt's protagonist

Every cell uses seed 0, so all settings, steps and models draw the same random numbers.

Usage:
    # temperature x top_k on one model
    python simple_stories_train/sample_study.py settings out/final_11M_600M/<timestamp>
    # the same setting across a run's intermediate checkpoints
    python simple_stories_train/sample_study.py steps out/final_11M_600M/<timestamp> --temperature 1
    # prompts with a protagonist, across models (run dirs or Hub ids)
    python simple_stories_train/sample_study.py prompts out/final_30M_600M/<timestamp> \
        SimpleStories/SimpleStories-30M
"""

import re
from collections import Counter
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import fire
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from simple_stories_train.evaluate import PACKAGE_DIR, load_hub_model, load_run
from simple_stories_train.models.llama import Llama

WORD = re.compile(r"[a-z]+")
TOKENIZER = Tokenizer.from_file(str(PACKAGE_DIR / "tokenizer/simplestories-4096.json"))
EOS = TOKENIZER.token_to_id("[EOS]")
SETTINGS = [(0.0, None), (0.5, None), (0.8, None), (1.0, None), (1.3, None), (1.6, None)] + [
    (1.0, 10),
    (1.0, 40),
    (1.3, 40),
]
STEPS = (0, 16, 64, 128, 256, 512, 1000, 2000, 4000, 8000)
# prompt -> words that name its protagonist
PROMPTS = {
    "the curious cat looked at the": {"cat"},
    "a little dog named max": {"max", "dog"},
    "the old fisherman": {"fisherman"},
    "a young girl named lily": {"lily"},
    "the brave knight": {"knight"},
}


@cache
def known_words() -> set[str]:
    train = load_dataset("lennart-finke/SimpleStories", split="train").select(range(200_000))
    return {w for story in train["story"] for w in WORD.findall(story.lower())}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(name: str, step: int | None, device: str) -> tuple[Llama, str]:
    if Path(name).exists():
        model, _, used_step = load_run(Path(name), step, device)
        return model, f"{name} (step {used_step})"
    return load_hub_model(name, device)[0], name


@torch.no_grad()
def sample(
    model: Llama,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
    prompt: str = "",
) -> list[list[int]]:
    """n samples from [EOS] + prompt, each cut at its first [EOS] (kept)."""
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False).ids
    idx = torch.tensor([[EOS, *prompt_ids]] * n, dtype=torch.long, device=device)
    torch.manual_seed(0)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with ctx:
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    rows = out[:, idx.shape[1] :].tolist()
    return [row[: row.index(EOS) + 1] if EOS in row else row for row in rows]


def has_loop(ids: list[int]) -> bool:
    grams = Counter(tuple(ids[i : i + 4]) for i in range(len(ids) - 3))
    return bool(grams) and max(grams.values()) >= 3


def metrics(rows: list[list[int]], protagonist: set[str] | None = None) -> dict[str, float]:
    texts = [TOKENIZER.decode(r) for r in rows]
    words = [WORD.findall(t) for t in texts]
    n_words = sum(map(len, words))
    result = {
        "invented": 100 * sum(w not in known_words() for ws in words for w in ws) / max(n_words, 1),
        "loops": 100 * sum(map(has_loop, rows)) / len(rows),
        "ended": 100 * sum(r[-1:] == [EOS] for r in rows) / len(rows),
    }
    if protagonist is not None:
        kept = sum(bool(protagonist & set(ws[len(ws) // 2 :])) for ws in words)
        result["keeps"] = 100 * kept / len(rows)
    return result


def report(label: str, m: dict[str, float], rows: list[list[int]], n_shown: int) -> None:
    cells = "  ".join(f"{k} {v:5.2f}%" for k, v in m.items())
    print(f"{label:<48} {cells}", flush=True)
    for r in rows[:n_shown]:
        print(f"    | {TOKENIZER.decode(r)[:300]}", flush=True)


def settings(
    model_name: str,
    n: int = 200,
    max_new_tokens: int = 200,
    n_shown: int = 2,
    device: str | None = None,
) -> None:
    """Temperature x top_k on the final checkpoint of one model."""
    device = device or default_device()
    model, label = load(model_name, None, device)
    print(f"{label}: {n} samples x {max_new_tokens} tokens from [EOS], seed 0\n")
    for temperature, top_k in SETTINGS:
        rows = sample(model, n, max_new_tokens, temperature, top_k, device)
        report(f"T={temperature} top_k={top_k}", metrics(rows), rows, n_shown)


def steps(
    run_dir: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    steps: tuple[int, ...] = STEPS,
    n: int = 200,
    max_new_tokens: int = 200,
    n_shown: int = 2,
    device: str | None = None,
) -> None:
    """One sampling setting across a run's intermediate checkpoints, then its final one."""
    device = device or default_device()
    print(f"T={temperature} top_k={top_k}: {n} samples x {max_new_tokens} tokens, seed 0\n")
    for step in [*steps, None]:
        model, label = load(run_dir, step, device)
        rows = sample(model, n, max_new_tokens, temperature, top_k, device)
        report(label, metrics(rows), rows, n_shown)


def prompts(
    *model_names: str,
    temperatures: tuple[float, ...] = (1.0, 0.8),
    n: int = 200,
    max_new_tokens: int = 300,
    device: str | None = None,
) -> None:
    """Prompts with a protagonist, across models; one row per prompt, then all prompts pooled."""
    device = device or default_device()
    print(f"{n} samples x {max_new_tokens} tokens per prompt, seed 0, no top_k\n")
    for name in model_names:
        model, label = load(name, None, device)
        for temperature in temperatures:
            pooled: Counter[str] = Counter()
            for prompt, protagonist in PROMPTS.items():
                rows = sample(model, n, max_new_tokens, temperature, None, device, prompt)
                m = metrics(rows, protagonist)
                pooled.update(m)
                report(f"T={temperature} {prompt!r}", m, rows, 0)
            mean = {k: v / len(PROMPTS) for k, v in pooled.items()}
            report(f"T={temperature} all prompts: {label}", mean, [], 0)
            print()


if __name__ == "__main__":
    fire.Fire({"settings": settings, "steps": steps, "prompts": prompts})

import torch
from datasets import Dataset
from tokenizers import Tokenizer

from simple_stories_train.dataloaders import (
    DatasetConfig,
    create_data_loader,
    tokenize_and_concatenate,
)


def test_tokenize_and_concatenate_puts_eos_only_between_stories():
    tokenizer = Tokenizer.from_file("simple_stories_train/tokenizer/simplestories-4096.json")
    eos_id = tokenizer.token_to_id("[EOS]")
    stories = [
        "The little girl ran home.",
        "A brave knight found a shiny sword in the cave.",
        "The stars twinkled above the quiet village at night.",
        "Tom and his dog played in the green park all day.",
    ]
    seq_len = 8

    out = tokenize_and_concatenate(
        Dataset.from_dict({"story": stories}), tokenizer, max_length=seq_len, num_proc=1
    )

    expected = []
    for story in stories:
        expected += tokenizer.encode(story.lower(), add_special_tokens=False).ids + [eos_id]
    expected = expected[: len(expected) // seq_len * seq_len]
    assert torch.cat(list(out["input_ids"])).tolist() == expected


def test_dataloader_with_ddp():
    # Load a small dataset from the datasets library
    dataset_config = DatasetConfig(
        name="lennart-finke/SimpleStories",
        is_tokenized=False,
        tokenizer_file_path="simple_stories_train/tokenizer/stories-3072.json",
        streaming=True,
        split="test",
        n_ctx=16,
        seed=42,
        column_name="story",
    )

    batch_size = 2
    buffer_size = 1000
    global_seed = 0

    # Create two DataLoaders with the same parameters and seed, to see they give the same data
    loader1, _ = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=0, ddp_world_size=1
    )
    loader2, _ = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=0, ddp_world_size=1
    )

    # Compare the first few batches
    num_batches_to_check = 5
    for i, (batch1, batch2) in enumerate(zip(loader1, loader2, strict=False)):
        assert batch1["input_ids"].shape == (batch_size, dataset_config.n_ctx)
        assert torch.equal(batch1["input_ids"], batch2["input_ids"]), f"Batch {i} is not the same"
        if i >= num_batches_to_check - 1:
            break

    #### Now test that data is different when ddp_rank is different
    # Create DataLoaders for rank 0 and rank 1
    loader_rank0, _ = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=0, ddp_world_size=2
    )
    loader_rank1, _ = create_data_loader(
        dataset_config, batch_size, buffer_size, global_seed, ddp_rank=1, ddp_world_size=2
    )

    # Compare the first few batches to ensure they are different
    batches_are_same = True
    for i, (batch0, batch1) in enumerate(zip(loader_rank0, loader_rank1, strict=False)):
        if not torch.equal(batch0["input_ids"], batch1["input_ids"]):
            batches_are_same = False
            break
        if i >= num_batches_to_check - 1:
            break

    assert not batches_are_same, "Batches are the same for different ddp_ranks"

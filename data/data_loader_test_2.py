def test_dataloader(dataset, dataloader):
    """Sanity-check an already constructed MusicDataset/DataLoader pair before training."""
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    masked_intervals, original_masks = next(iter(dataloader))
    print(f"Masked intervals shape: {masked_intervals.shape}")
    print(f"Original masks shape: {original_masks.shape}")

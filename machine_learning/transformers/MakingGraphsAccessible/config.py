class CONFIG:
    debug = False
    accelerator = "cpu" if debug else "gpu"
    devices = "auto" if accelerator == "cpu" else [1]
    batch_size = 2 if debug else 1
    limit_train_batches = 2 if debug else None
    limit_val_batches = 2 if debug else 100
    learning_rate = 3e-5
    val_fraction = 0.1
    seed = 42
    train_val_indices_path = "data/train_val_indices.pickle"
    float_scientific_notation_string_precision = 5
    pretrained_model_name = "naver-clova-ix/donut-base"
    image_width = 720
    image_height = 512
    unknown_tokens_for_tokenizer_path = "data/unknown_tokens_for_tokenizer.pickle"
    decoder_sequence_max_length = 512
    num_workers = 4
    training_directory = "training"
    save_top_k_checkpoints = 3
    wandb_project_name = "MakingGraphsAccessible"

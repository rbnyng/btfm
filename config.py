# config.py

default_config = {
    # General settings
    "project_name": "btfm",
    "experiment_name": "default_experiment",
    "checkpoint_dir": "checkpoints/",
    
    # Model settings
    "backbone": "transformer",  # options: transformer, simple_mlp, simple_cnn, resnet
    "backbone_param_hidden_dim": 128,
    "backbone_param_num_layers": 2,
    "latent_dim": 64,
    "projection_head_hidden_dim": 128,
    "projection_head_output_dim": 128,
    "time_dim": 0,
    
    # Training settings
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 0.0001,
    # "learning_rate": 0.00001,
    # "warmup_steps": 5000,
    "warmup_steps": 1000,
    "warmdown_steps": 10000,
    "barlow_lambda": 5e-4,
    "mmcr_lambda": 5e-4,
    "mmcr_alpha": 5e-3,
    "weight_decay": 1.5e-6,
    "gradient_clipping": 1.0,
    
    # Dataset settings
    "data_dir": "../../../maps-priv/maps/ray25/", # Path to the SSL training/validation data
    "base_downstream_path": "../btfm-data-preparation/", # Path to the downstream data
    "test_tile_path": "/maps-priv/maps/ray25/data/germany/processed/MGRS-32UPB", # Path to the test tile
    "train_dataset": "germany", # options: wyoming, california
    "val_dataset": "germany",
    "min_valid": 32,
    "sample_size": 16,
    "band_size": 10,
    # "validation_size": 65536 * 4,
    "validation_size": 65536 * 8,
    # "validation_size": 10000,
    
    # DataLoader settings
    "num_workers": 6,

    # Matryoshka settings
    "nesting_dims": [32, 64, 128, 256, 512],  # Dimensions for nested representations
    "nesting_weights": [0.5, 0.75, 1.0, 1.0, 1.0],  # Optional weights for each dimension
    "backbone": "matryoshka_transformer",  # New backbone option
    "projector_dims": [512, 512, 512, 512, 512],  # Output dims for each granularity
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.2,
    "input_dim": 10, 
    
    # Logging and visualization
    "log_interval_steps": 64,
    "val_interval_steps": 8192,
    # "val_interval_steps": 1000,
    "img_interval_steps": 8192 * 4,
    
    # Hardware settings
    "device": "cuda",  # options: cuda, cpu
    
    # Commit link
    "commit_link": "" # initialize with empty string
}

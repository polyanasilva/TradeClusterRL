CONFIG = {
    "data": {
        "raw_path": "data/raw",
        "processed_path": "data/processed"
    },
    "clustering": {
        "n_clusters": 175,
        "scaler": "standard"
    },
    "rl": {
        "gamma": 0.99,
        "batch_size": 64,
        "episodes": 500,
        "max_steps": 1000,
        "learning_rate": 1e-4
    } 
}
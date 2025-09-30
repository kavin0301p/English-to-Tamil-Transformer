from pathlib import Path
    
def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 2e-4,
        "seq_len": 128,
        "d_model": 360,
        "lang_src": "en",
        "lang_tgt": "ta",
        "model_folder": "weights",
        "model_basename": "/kaggle/working/weights",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",          
        "experiment_name": "runs/tmodel",
        "N": 6,
        "h": 8,
        "dropout": 0.2
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

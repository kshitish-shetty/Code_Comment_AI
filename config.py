from pathlib import Path

def get_config():
    return {
        "d_ff": 2048,
        "batch_size": 8,
        "num_layers": 3,
        "num_epochs": 20,
        "lr": 1e-4,
        "code_seq_len": 200,
        "cat_seq_len": 200,
        "max_seq_len": 400,
        "comment_seq_len": 30,
        "d_model": 256,
        "train_datasource":'myDataset/train.json',
        "test_datasource":'myDataset/test.json',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) 

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import get_config, latest_weights_file_path, get_weights_file_path
from tokenizer import get_or_build_tokenizer
from dataset import ALSITransformerDataset
from model2 import Transformer
from test import run_validation

import warnings
from torch.utils.tensorboard import SummaryWriter

def get_ds(config):
     # Load the training data from JSON file
    with open(config['train_datasource'], 'r') as f:
        data = json.load(f)

    with open(config['test_datasource'], 'r') as f:
        test_data = json.load(f)

    # Build tokenizers
    tokenizer_code = get_or_build_tokenizer(config, data, "code")
    tokenizer_cat = get_or_build_tokenizer(config, data, "CAT")
    tokenizer_comment = get_or_build_tokenizer(config, data, "comment")

    max_len_code = 0
    max_len_cat = 0
    max_len_comment = 0
    count = 0

    # for item in tqdm(data, desc="Processing items", unit="item"):
    #   code = tokenizer_code.encode(item['code']).ids
    #   cat = tokenizer_cat.encode(item['CAT']).ids
    #   comment = tokenizer_comment.encode(item['comment']).ids

    #   count += 1 if (len(code) > 200 or len(cat) > 200 or len(comment) > 30) else 0

    #   max_len_code = max(max_len_code, len(code))
    #   max_len_cat = max(max_len_cat, len(cat))
    #   max_len_comment = max(max_len_comment, len(comment))

    # print(f'Max length of code sentance: {max_len_code}')
    # print(f'Max length of CAT sentance: {max_len_cat}')
    # print(f'Max length of comment sentance: {max_len_comment}')
    # print(f'Number of data greater that seq_len: {count}')

    # Create dataset
    dataset = ALSITransformerDataset(data, tokenizer_code, tokenizer_cat, tokenizer_comment, config['code_seq_len'], config['comment_seq_len'], "Train")
    test_dataset = ALSITransformerDataset(test_data, tokenizer_code, tokenizer_cat, tokenizer_comment, config['code_seq_len'], config['comment_seq_len'], "Test")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    validation_ds = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return dataloader, validation_ds, tokenizer_code, tokenizer_cat, tokenizer_comment

def get_model(config, code_vocab_size, cat_vocab_size, comment_vocab_size):
    model = Transformer(code_vocab_size, cat_vocab_size, comment_vocab_size, config['d_model'],config['batch_size'], config['num_layers'], config['d_ff'], config['max_seq_len'], dropout=0.2)
    return model

def train_model(config):

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    dataloader, validation_ds, tokenizer_code, tokenizer_cat, tokenizer_comment = get_ds(config)
    model = get_model(config, tokenizer_code.get_vocab_size(), tokenizer_cat.get_vocab_size(), tokenizer_comment.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_code.token_to_id('[PAD]'), label_smoothing=0.0).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input_code = batch['encoder_input_code'].to(device)
            encoder_input_cat = batch['encoder_input_cat'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask_code = batch['encoder_mask_code'].to(device)
            encoder_mask_cat = batch['encoder_mask_cat'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            model_output = model(encoder_input_code, encoder_input_cat, encoder_mask_code, encoder_mask_cat, decoder_input, decoder_mask)

            label = batch['label'].to(device)

            loss = loss_fn(model_output.view(-1, tokenizer_comment.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, validation_ds, tokenizer_comment, config['comment_seq_len'], device, lambda msg: batch_iterator.write(msg))
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
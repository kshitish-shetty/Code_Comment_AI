import json
from pathlib import Path
import sys
from config import get_config, latest_weights_file_path 
from model2 import Transformer
from tokenizers import Tokenizer
from dataset import ALSITransformerDataset, casual_mask
from get_CAT import generate_code_aligned_type_sequence
import torch

def translate(code_input: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)
    config = get_config()
    tokenizer_code = Tokenizer.from_file(str(Path(config['tokenizer_file'].format("code"))))
    tokenizer_cat = Tokenizer.from_file(str(Path(config['tokenizer_file'].format("CAT"))))
    tokenizer_comment = Tokenizer.from_file(str(Path("tokenizer_comment.json")))
    model = Transformer(tokenizer_code.get_vocab_size(), tokenizer_cat.get_vocab_size(), tokenizer_comment.get_vocab_size(), config['d_model'],config['batch_size'], config['num_layers'], config['d_ff'], config['max_seq_len'], dropout=0.2)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    comment_input = ""
    cat_input = ""
    if type(code_input) == int or code_input.isdigit():
        id = int(code_input)
        with open(config['train_datasource'], 'r') as f:
            ds = json.load(f)
        ds = ALSITransformerDataset(ds, tokenizer_code, tokenizer_cat, tokenizer_comment, config['code_seq_len'], config['comment_seq_len'], "Train")
        code_input = ds[id]['code_text']
        cat_input = ds[id]['cat_text']
        comment_input = ds[id]["comment_text"]
    code_seq_len = config['code_seq_len']
    cat_seq_len = config['cat_seq_len']
    comment_seq_len = config['comment_seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        encoder_code_input = tokenizer_code.encode(code_input)
        encoder_code_input = torch.cat([
            torch.tensor([tokenizer_code.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(encoder_code_input.ids, dtype=torch.int64),
            torch.tensor([tokenizer_code.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_code.token_to_id('[PAD]')] * (code_seq_len - len(encoder_code_input.ids) - 2), dtype=torch.int64)
        ]).unsqueeze(0).to(device)
        encoder_code_mask = (encoder_code_input != tokenizer_code.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).to(device)

        if cat_input == "": 
            cat_input = generate_code_aligned_type_sequence(code_input)

        encoder_cat_input = tokenizer_cat.encode(cat_input)
        encoder_cat_input = torch.cat([
            torch.tensor([tokenizer_cat.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(encoder_cat_input.ids, dtype=torch.int64),
            torch.tensor([tokenizer_cat.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_cat.token_to_id('[PAD]')] * (cat_seq_len - len(encoder_cat_input.ids) - 2), dtype=torch.int64)
        ]).unsqueeze(0).to(device)
        encoder_cat_mask = (encoder_cat_input != tokenizer_cat.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).to(device)

        enc_output, src_mask = model.encode(encoder_code_input, encoder_cat_input, encoder_code_mask, encoder_cat_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_comment.token_to_id('[SOS]')).type_as(encoder_code_input).to(device)

        # Print the source sentence and target start prompt
        # if comment_input != "": print(f"{f'ID: ':>12}{id}") 
        # print(f"{f'CODE: ':>12}{code_input}")
        # if comment_input != "": print(f"{f'COMMENT: ':>12}{comment_input}") 
        # print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < comment_seq_len:
            # build mask for target and calculate output
            decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_code_mask).to(device)
            out = model.decode(decoder_input, enc_output, src_mask, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_code_input).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            # print(f"{tokenizer_comment.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_comment.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_comment.decode(decoder_input[0].tolist())
    
#read sentence from argument
# translate(sys.argv[1] if len(sys.argv) > 1 else "public boolean isUpdateRequired ( ) { return this . updateRequired ; }")
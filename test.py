import os
import torch
from dataset import casual_mask

def greedy_decode(model, encoder_input_code, encoder_input_cat, encoder_mask_code, encoder_mask_cat, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    enc_output, src_mask = model.encode(encoder_input_code, encoder_input_cat, encoder_mask_code, encoder_mask_cat)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input_code).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_mask_code).to(device)

        # calculate output
        out = model.decode(decoder_input, enc_output, src_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(encoder_input_code).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_comment, max_len, device, print_msg, num_examples=2):

    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input_code = batch['encoder_input_code'].to(device)
            encoder_input_cat = batch['encoder_input_cat'].to(device)

            encoder_mask_code = batch['encoder_mask_code'].to(device)
            encoder_mask_cat = batch['encoder_mask_cat'].to(device)

            # check that the batch size is 1
            assert encoder_input_code.size(
                0) == 1, "Code Batch size must be 1 for validation"
            assert encoder_input_cat.size(
                0) == 1, "CAT Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input_code, encoder_input_cat, encoder_mask_code, encoder_mask_cat, tokenizer_comment, max_len, device)

            code = batch["code_text"]
            comment = batch["comment_text"]
            model_out_comment = tokenizer_comment.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'CODE: ':>12}{code}")
            print_msg(f"{f'COMMENT: ':>12}{comment}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_comment}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # if writer:
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()
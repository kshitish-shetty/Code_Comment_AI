import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ALSITransformerDataset(Dataset):
    def __init__(self, ds, tokenizer_code, tokenizer_cat, tokenizer_comment, code_seq_len, comment_seq_len, TrainOrTest):
        super().__init__()
        self.tokenizer_code = tokenizer_code
        self.tokenizer_cat = tokenizer_cat
        self.tokenizer_comment = tokenizer_comment
        self.code_seq_len = code_seq_len
        self.comment_seq_len = comment_seq_len
        
        self.sos_token = torch.tensor([tokenizer_code.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_code.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_code.token_to_id('[PAD]')], dtype=torch.int64)

        removed_count = 0  # Counter for removed samples

        filtered_ds = []
        for sample in tqdm(ds, desc=f"Loading {TrainOrTest} Dataset", total=len(ds)):
            # Tokenize and truncate if necessary
            code_tokens = tokenizer_code.encode(sample['code']).ids[:code_seq_len - 2]
            cat_tokens = tokenizer_cat.encode(sample['CAT']).ids[:code_seq_len - 2]
            comment_tokens = tokenizer_comment.encode(sample['comment']).ids[:comment_seq_len - 1]

            filtered_ds.append({
                'code': code_tokens,
                'CAT': cat_tokens,
                'comment': comment_tokens
            })

        self.ds = filtered_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        code = sample['code']
        cat = sample['CAT']
        comment = sample['comment']

        # Encode sequences
        enc_input_code = code
        enc_input_cat = cat
        dec_input_comment = comment

        # Padding calculations
        enc_padding_code = self.code_seq_len - len(enc_input_code) - 2
        enc_padding_cat = self.code_seq_len - len(enc_input_cat) - 2
        dec_padding_comment = self.comment_seq_len - len(dec_input_comment) - 1

        if enc_padding_code < 0 or enc_padding_cat < 0 or dec_padding_comment < 0:
            raise ValueError('Input sequences are too long')

        # Encoder input: combine code and CAT through the gate network
        encoder_input_code = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_code, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_padding_code, dtype=torch.int64)
        ])

        encoder_input_cat = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_cat, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_padding_cat, dtype=torch.int64)
        ])

        # Decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_comment, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_padding_comment, dtype=torch.int64)
        ])

        # Label for training
        label = torch.cat([
            torch.tensor(dec_input_comment, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_padding_comment, dtype=torch.int64)
        ])

        assert encoder_input_code.size(0) == self.code_seq_len
        assert encoder_input_cat.size(0) == self.code_seq_len
        assert decoder_input.size(0) == self.comment_seq_len
        assert label.size(0) == self.comment_seq_len

        return {
            "encoder_input_code": encoder_input_code,
            "encoder_input_cat": encoder_input_cat,
            "decoder_input": decoder_input,
            "encoder_mask_code": (encoder_input_code != self.pad_token).unsqueeze(0).unsqueeze(0),
            "encoder_mask_cat": (encoder_input_cat != self.pad_token).unsqueeze(0).unsqueeze(0),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) & casual_mask(decoder_input.size(0)),
            "label": label,
            "code_text": self.tokenizer_code.decode(code),
            "cat_text": self.tokenizer_cat.decode(cat),
            "comment_text": self.tokenizer_comment.decode(comment)
        }

def casual_mask(size):
    mask = (1 - torch.triu(torch.ones(1, size, size), diagonal=1)).bool()
    return mask
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

def get_all_sentences(ds, key):
    """Yields all sentences in the dataset for a given key ('code', 'CAT', or 'comment')."""
    for item in ds:
        yield item[key]

def get_or_build_tokenizer(config, ds, key):
    tokenizer_path = Path(config['tokenizer_file'].format(key))

    if not tokenizer_path.exists():

        if key == 'comment':
            path = "myDataset/train.token.nl"
        
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=52000,
                show_progress = True,
                special_tokens = ["[PAD]","[UNK]", "[SOS]", "[EOS]"])
            tokenizer.train(files=[path], trainer=trainer)

            tokenizer.save(str(tokenizer_path))

        else:
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
            tokenizer.train_from_iterator(get_all_sentences(ds, key), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

from model import EncoderDecoder
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from torchtext.legacy.data import BucketIterator, Field

from utilities import DEVICE, load_dataset, load_word_field, DATA_PATH, BOS_TOKEN, EOS_TOKEN
from optimizer import NoamOpt
from train import fit

def main():
    word_field = load_word_field(DATA_PATH)
    train_dataset = load_dataset(DATA_PATH + "/train/")
    test_dataset = load_dataset(DATA_PATH + "/test/")

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
    )  
    
    tqdm.get_lock().locks = []
    
    model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab)).to(DEVICE)

    pad_idx = word_field.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(DEVICE)

    optimizer = NoamOpt(model.d_model, model)

    fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=test_iter)
    
    
if __name__ == '__main__':
    main()
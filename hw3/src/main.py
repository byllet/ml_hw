from model import EncoderDecoder
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from torchtext.legacy.data import BucketIterator
import wandb
import time

from utilities import DEVICE, load_dataset, load_word_field
from params import DATA_PATH, config
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
    loading = False
    with_prelearning_emb = False
    model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab),
                           with_prel_emb=with_prelearning_emb).to(DEVICE)

    if loading:
        model.load_state_dict(torch.load('models/model_temp.pt'))
    model.to(DEVICE)

    pad_idx = word_field.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config['label_smoothing']).to(DEVICE)

    optimizer = NoamOpt(model.d_model, model)

    fit(model, criterion, optimizer, train_iter, start_epoch=0, epochs_count=config['epochs'], val_iter=test_iter)
    torch.save(model.state_dict(), "models/model_trained.pt")
    
    
if __name__ == '__main__':
    main()
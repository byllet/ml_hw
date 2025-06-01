import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RANDOM_SEED
from encoder import Encoder
from decoder import Decoder
from utilities import PositionalEncoding
from navec import Navec
from slovnet.model.emb import NavecEmbedding

np.random.seed(RANDOM_SEED)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, source_vocab_size, target_vocab_size, with_prel_emb, d_model=300, d_ff=1024, 
                 blocks_count=4, heads_count=6, dropout_rate=0.1):
        super(EncoderDecoder, self).__init__()

        self.d_model = d_model
        _path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'  # 51MB
        _navec = Navec.load(_path)
        
        if with_prel_emb:
            self._emb = nn.Sequential(
                NavecEmbedding(_navec),
                PositionalEncoding(d_model, dropout_rate)
            )
        else:
            self._emb = nn.Sequential(
                nn.Embedding(source_vocab_size, d_model),
                PositionalEncoding(d_model, dropout_rate)
            )
        self.encoder = Encoder(d_model, d_ff, 
                               blocks_count, heads_count, dropout_rate, self._emb)
        self.decoder = Decoder(target_vocab_size, d_model, d_ff, 
                               blocks_count, heads_count, dropout_rate, self._emb)
        # self.generator = Generator(d_model, target_vocab_size)

        for p in self.parameters():
            if type(p) is torch.float and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        encoder_output = self.encoder(source_inputs, source_mask)
        return self.decoder(target_inputs, encoder_output, source_mask, target_mask)
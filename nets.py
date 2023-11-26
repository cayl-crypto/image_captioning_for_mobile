import torch
from torch._C import device
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ICT(nn.Module):
    def __init__(self, vocab_len, *args, **kwargs) -> None:
        super(ICT, self).__init__(*args, **kwargs)

        self.tgt_embedding = nn.Embedding(vocab_len, 768)
        self.src_pe = PositionalEncoding(d_model=768)
        self.tgt_pe = PositionalEncoding(d_model=768)
        self.transformer = nn.Transformer(d_model=768, batch_first=True)
        self.generator = nn.Linear(768, vocab_len)

    def forward(self, src, tgt):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pe(tgt)
        src = self.src_pe(src)
        x = self.transformer(src, tgt)
        x = self.generator(x)
        return x

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.src_pe(src), src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        tgt_input = self.tgt_pe(self.tgt_embedding(tgt))
        return self.transformer.decoder(tgt_input, memory, tgt_mask)

class GNMT(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, embed_size, device):
        super(GNMT, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru_1 = nn.GRU(embed_size, hidden_size)
        self.gru_2 = nn.GRU(hidden_size, hidden_size)
        self.gru_3 = nn.GRU(hidden_size, hidden_size)
        self.gru_4 = nn.GRU(hidden_size, hidden_size)
        self.gru_5 = nn.GRU(hidden_size, hidden_size)
        self.gru_6 = nn.GRU(hidden_size, hidden_size)
        self.gru_7 = nn.GRU(hidden_size, hidden_size)
        self.gru_8 = nn.GRU(hidden_size, hidden_size)

        self.map = nn.Linear(feature_size, hidden_size)
        self.attn_input = nn.Linear(hidden_size, hidden_size)
        self.attn_hidden = nn.Linear(hidden_size, hidden_size)
        self.attn_output = nn.Linear(hidden_size, hidden_size)


        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def first_step(self, feature, x, hidden):
        x = F.relu(self.embedding(x))
        x = x.unsqueeze(0)
        attn_input, hidden_1 = self.gru_1(x, hidden)
        feature = self.map(feature)
        x2, hidden_2 = self.gru_2(torch.add(feature, attn_input), hidden)
        x3, hidden_3 = self.gru_3(x2, hidden)
        x4, hidden_4 = self.gru_4(torch.add(x2, x3), hidden)
        x5, hidden_5 = self.gru_5(torch.add(x3, x4), hidden)
        x6, hidden_6 = self.gru_6(torch.add(x4, x5), hidden)
        x7, hidden_7 = self.gru_7(torch.add(x5, x6), hidden)
        x8, hidden_8 = self.gru_8(torch.add(x6, x7), hidden)
        x = self.softmax(self.out(x8[0]))
        hiddens = [hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6, hidden_7, hidden_8]
        return x, attn_input, hiddens

    def forward(self, feature, x, attention, hiddens):
        x = F.relu(self.embedding(x))
        x = x.unsqueeze(0)
        feature = feature.unsqueeze(0)
        feature = F.relu(self.map(feature))
        attn = F.relu(self.attn_input(torch.add(feature, attention)))
        attn = F.relu(self.attn_hidden(attn))
        attn = F.relu(self.attn_output(attn))


        attention, hidden_1 = self.gru_1(x, torch.add(hiddens[0], attn))
        x2, hidden_2 = self.gru_2(attention, torch.add(hiddens[1], attn))
        x3, hidden_3 = self.gru_3(x2, torch.add(hiddens[2], attn))
        x4, hidden_4 = self.gru_4(torch.add(x2, x3), torch.add(hiddens[3], attn))
        x5, hidden_5 = self.gru_5(torch.add(x3, x4), torch.add(hiddens[4], attn))
        x6, hidden_6 = self.gru_6(torch.add(x4, x5), torch.add(hiddens[5], attn))
        x7, hidden_7 = self.gru_7(torch.add(x5, x6), torch.add(hiddens[6], attn))
        x8, hidden_8 = self.gru_8(torch.add(x6, x7), torch.add(hiddens[7], attn))

        hiddens = [hidden_1, hidden_2, hidden_3, hidden_4, hidden_5, hidden_6, hidden_7, hidden_8]
        x = self.softmax(self.out(x8[0]))

        return x, attention, hiddens

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)

        output = self.embedding(input)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)

        output = self.embedding(input)

        output = F.relu(output)

        output, hidden = self.lstm(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden


import torch
import torch.nn as nn

class LSTMWithCacheAndChar(nn.Module):
    def __init__(self, word_vocab_size,
                 char_vocab_size,
                 word_embed_dim,
                 char_embed_dim,
                 hidden_dim,
                 char_hidden_dim,
                 num_layers,
                 cache_size,
                 max_word_len):
        super(LSTMWithCacheAndChar, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)

        self.char_lstm = nn.LSTM(char_embed_dim, char_hidden_dim, batch_first=True)

        self.lstm = nn.LSTM(word_embed_dim + char_hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, word_vocab_size)

        self.cache = {}
        self.cache_size = cache_size

    def forward(self, word_inputs, char_inputs):
        batch_size, seq_len, max_word_len = char_inputs.size()
        char_inputs = char_inputs.view(-1, max_word_len)
        char_embeds = self.char_embedding(char_inputs)
        _, (char_hidden, _) = self.char_lstm(char_embeds)
        char_hidden = char_hidden[-1].view(batch_size, seq_len, -1)

        word_embeds = self.word_embedding(word_inputs)
        combined_embeds = torch.cat((word_embeds, char_hidden), dim=-1)

        lstm_out, _ = self.lstm(combined_embeds)
        logits = self.fc(lstm_out)

        for idx in range(batch_size):
            seq_key = word_inputs[idx].cpu().numpy().tostring()
            if seq_key not in self.cache:
                self.cache[seq_key] = logits[idx]
                if len(self.cache) > self.cache_size:
                    self.cache.pop(next(iter(self.cache)))

        return logits

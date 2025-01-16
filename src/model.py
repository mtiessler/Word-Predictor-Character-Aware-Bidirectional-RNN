import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMWithCacheAndChar(nn.Module):
    def __init__(self, word_vocab_size,
                 char_vocab_size,
                 word_embed_dim,
                 char_embed_dim,
                 hidden_dim,
                 char_hidden_dim,
                 num_layers,
                 cache_size):
        super(LSTMWithCacheAndChar, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)

        # Bidirectional LSTMs
        self.char_lstm = nn.LSTM(char_embed_dim, char_hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(word_embed_dim + 2 * char_hidden_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(2 * hidden_dim, word_vocab_size)  # Adjusted for bidirection

        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(0.5)  # Dropout rate can be adjusted
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)  # Adjusted for bidirection

        # Cache for storing recent sequences
        self.cache = {}
        self.cache_size = cache_size

        # Regularization strength (adjust as needed)
        self.l2_lambda = 1e-5

    def forward(self, word_inputs, char_inputs):
        batch_size, seq_len, max_word_len = char_inputs.size()
        char_inputs = char_inputs.view(-1, max_word_len)

        # Character embedding and LSTM
        char_embeds = self.char_embedding(char_inputs)
        char_embeds = self.dropout(char_embeds)
        _, (char_hidden, _) = self.char_lstm(char_embeds)
        char_hidden = char_hidden[-2:].permute(1, 0, 2).contiguous().view(batch_size, seq_len, -1)

        # Word embedding and combination
        word_embeds = self.word_embedding(word_inputs)
        word_embeds = self.dropout(word_embeds)
        combined_embeds = torch.cat((word_embeds, char_hidden), dim=-1)

        # Main LSTM processing
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.layer_norm(lstm_out)

        # Fully connected layer for prediction
        logits = self.fc(lstm_out)

        # Normalize logits for numerical stability
        logits = F.log_softmax(logits, dim=-1)

        return logits
    def l2_regularization(self):
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param.pow(2))
        return self.l2_lambda * l2_loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, code_vocab_size, cat_vocab_size, comment_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_code_embedding = nn.Embedding(code_vocab_size, d_model)
        self.encoder_CAT_embedding = nn.Embedding(cat_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(comment_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.code_CNN = nn.Conv2d(1, 1, (3, 1), padding=(1, 0))
        self.CAT_CNN = nn.Conv2d(1, 1, (3, 1), padding=(1, 0))

        self.gate_layer = nn.Linear(d_model * 2, d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, comment_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, code, CAT, code_mask, CAT_mask):
        code_embedded = self.dropout(self.positional_encoding(self.encoder_code_embedding(code)))
        code_embedded = code_embedded.unsqueeze(1)  # Shape: (N, 1, code_len, hidden_size)
        code_embedded = self.code_CNN(code_embedded)
        code_embedded = F.max_pool2d(code_embedded, (2, 1), stride=(2, 1))
        code_embedded = code_embedded.squeeze(1)

        CAT_embedded = self.dropout(self.positional_encoding(self.encoder_CAT_embedding(CAT)))
        CAT_embedded = CAT_embedded.unsqueeze(1)  # Shape: (N, 1, code_len, hidden_size)
        CAT_embedded = self.CAT_CNN(CAT_embedded)
        CAT_embedded = F.max_pool2d(CAT_embedded, (2, 1), stride=(2, 1))
        CAT_embedded = CAT_embedded.squeeze(1)

        AB_concat = torch.cat((code_embedded, CAT_embedded), dim=-1)  # Shape: (N, code_len // 2, hidden_size * 2)
        context_gate = torch.sigmoid(self.gate_layer(AB_concat))
        src_embedded = context_gate * code_embedded + (1 - context_gate) * CAT_embedded

        code_mask = code_mask[:, :, :, ::2]  # After pooling: Shape (N, 1, 1, code_len // 2)
        CAT_mask = CAT_mask[:, :, :, ::2]    # After pooling: Shape (N, 1, 1, sbt_len // 2)

        # Mask
        src_mask = torch.logical_or(code_mask, CAT_mask)

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        return enc_output, src_mask
    
    def decode(self, comment, enc_output, src_mask, comment_mask):
        comment_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(comment)))
        dec_output = comment_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, comment_mask)
        return dec_output
    
    def project(self, decoder_output):
        return self.fc(decoder_output)

    def forward(self, code, CAT, code_mask, CAT_mask, comment, comment_mask):

        enc_output, src_mask = self.encode(code, CAT, code_mask, CAT_mask)

        dec_output = self.decode(comment, enc_output, src_mask, comment_mask)

        # weights = self.decoder_embedding.weight.t()   # Shape: (d_model, vocab_size)
        # logits = torch.einsum('ntd,dk->ntk', dec_output, weights)  # Shape: (N, T2, vocab_size)
        # output = torch.argmax(logits, dim=-1).to(torch.int32)

        output = self.fc(dec_output)
        return output
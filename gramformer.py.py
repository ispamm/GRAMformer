import math
import torch
import torch.nn.functional as F
from torch import nn



    
class TransformerEncoder_gram(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers.
    
    Each layer is a :class:`TransformerEncoderLayer`.
    
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        """Initialize Transformer Encoder.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of heads
            layers (int): Number of layers
            attn_dropout (float, optional): Probability of dropout in attention mechanism. Defaults to 0.0.
            relu_dropout (float, optional): Probability of dropout after ReLU. Defaults to 0.0.
            res_dropout (float, optional): Probability of dropout in residual layer. Defaults to 0.0.
            embed_dropout (float, optional): Probability of dropout in embedding layer. Defaults to 0.0.
            attn_mask (bool, optional): Whether to apply a mask to the attention or not. Defaults to False.
        """
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            new_layer = TransformerEncoderLayer_GRAM(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, mod1_q, mod2=None, mod3=None, return_attn=False):
        """
        Apply Transformer Encoder to layer input.
        
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        #print("mod1_q shape", mod1_q.shape)
        #print("mod2 shape", mod2.shape if mod2 is not None else None)
        #print("mod3 shape", mod3.shape if mod3 is not None else None)
        # embed tokens and positions
        x = self.embed_scale * mod1_q
        if self.embed_positions is not None:
            # Add positional embedding
            x += self.embed_positions(mod1_q.transpose(0, 1)
                                      [:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if mod2 is not None and mod3 is not None:
            # embed tokens and positions
            mod2 = self.embed_scale * mod2
            mod3 = self.embed_scale * mod3
            if self.embed_positions is not None:
                # Add positional embedding
                mod2 += self.embed_positions(mod2.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
                # Add positional embedding
                mod3 += self.embed_positions(mod3.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
            mod2 = F.dropout(mod2, p=self.dropout, training=self.training)
            mod3 = F.dropout(mod3, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        attention_maps = []
        for layer in self.layers:
            if mod2 is not None and mod3 is not None:
                if return_attn:
                    x, layer_attn = layer(x, mod2, mod3, return_attn=True)
                    attention_maps.append(layer_attn)
                else:
                    x = layer(x, mod2, mod3)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        if return_attn:
            return x, attention_maps
        return x   

   


def compute_attention_scores_parallel_gram(query, key_1, key_2, eps=1e-8):
    """
    Compute 3x3 Gram volumes for all pairs of language_i with (video_j, audio_j).

    query: [B, N_lang, D]
    key_1:    [B, N_vid,  D]
    key_2:    [B, N_vid,  D]

    Returns:
        volume: [B, N_lang, N_vid]
    """

    B, N_lang, D = query.shape
    N_vid = key_1.shape[1]

    # Expand for broadcasting
    l = query[:, :, None, :]   # [B, N_lang, 1, D]
    v = key_1[:, None, :, :]      # [B, 1, N_vid, D]
    a = key_2[:, None, :, :]      # [B, 1, N_vid, D]

    # Pairwise dot products
    ll = (l * l).sum(-1).expand(-1, -1, N_vid)      # [B, N_lang, N_vid]
    vv = (v * v).sum(-1).expand(-1, N_lang, -1)
    aa = (a * a).sum(-1).expand(-1, N_lang, -1)

    lv = (l * v).sum(-1)                            # [B, N_lang, N_vid]
    la = (l * a).sum(-1)
    va = (v * a).sum(-1).expand(-1, N_lang, -1)

    # Analytical determinant of Gram matrix
    det = (
        ll * (vv * aa - va * va)
        - lv * (lv * aa - la * va)
        + la * (lv * va - la * vv)
    )

    return -torch.sqrt(torch.clamp(det, min=eps))


class CustomMultiheadAttention_GRAM(nn.Module):
    """Custom implementation of Multi-Head Attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        """Initialize CustomMultiheadAttention.
        
        Args:
            embed_dim (int): Total dimension of the model.
            num_heads (int): Number of parallel attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bias (bool, optional): If True, add bias to input/output projection layers. Defaults to True.
            add_bias_kv (bool, optional): If True, add bias to the key and value sequences at dim=0. Defaults to False.
            add_zero_attn (bool, optional): If True, add a new batch of zeros to key and value sequences. Defaults to False.
            kdim (int, optional): Total number of features for keys. Defaults to None (uses embed_dim).
            vdim (int, optional): Total number of features for values. Defaults to None (uses embed_dim).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k1_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.k2_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v1_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.v2_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.v1_out_gate = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.v2_out_gate = nn.Linear(self.vdim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, query, mod1, mod2, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, return_score_components=False):
        """Forward pass of multi-head attention.
        
        Args:
            query (Tensor): Query tensor of shape (L, N, E) where L is target sequence length,
                          N is batch size, E is embedding dimension.
            key (Tensor): Key tensor of shape (S, N, E) where S is source sequence length.
            value (Tensor): Value tensor of shape (S, N, E).
            key_padding_mask (Tensor, optional): Mask for key padding of shape (N, S).
            need_weights (bool, optional): If True, return attention weights. Defaults to True.
            attn_mask (Tensor, optional): Mask for attention of shape (L, S) or (N*num_heads, L, S).
            average_attn_weights (bool, optional): If True, return averaged attention weights. Defaults to True.
            
        Returns:
            Tuple[Tensor, Tensor]: Output tensor of shape (L, N, E) and attention weights.
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = mod1.size(0)
        
        # Project Q, K, V
        q = self.q_proj(query)  # (L, N, E)
        k1 = self.k1_proj(mod1)    # (S, N, E)
        k2 = self.k2_proj(mod2)    # (S, N, E)
        v1 = self.v1_proj(mod1)    # (S, N, E)
        v2 = self.v2_proj(mod2)    # (S, N, E)
        
        # Scale query
        q = q * self.scaling
        
        # Reshape for multi-head attention
        # (L, N, E) -> (L, N, num_heads, head_dim) -> (N, num_heads, L, head_dim) -> (N*num_heads, L, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k1 = k1.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k2 = k2.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v1 = v1.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v2 = v2.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        # (N*num_heads, L, head_dim) x (N*num_heads, head_dim, S) -> (N*num_heads, L, S)
        attn_output_weights1 = torch.bmm(q, k1.transpose(1, 2))
        attn_output_weights2 = torch.bmm(q, k2.transpose(1, 2))
        attn_output_weights_gram = compute_attention_scores_parallel_gram(q, k1, k2)
        attn_output_weights_logits = ( 1.5 * attn_output_weights_gram) + attn_output_weights1 + attn_output_weights2
        attn_output_weights = attn_output_weights_logits
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                attn_output_weights += attn_mask
                
            elif attn_mask.dim() == 3:
                attn_output_weights += attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

            
        # Apply softmax
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        # (N*num_heads, L, S) x (N*num_heads, S, head_dim) -> (N*num_heads, L, head_dim)
        attn_output1 = torch.bmm(attn_output_weights, v1)
        attn_output2 = torch.bmm(attn_output_weights, v2)
        
        # Reshape back
        # (N*num_heads, L, head_dim) -> (L, N*num_heads, head_dim) -> (L, N, E)
        attn_output1 = attn_output1.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output2 = attn_output2.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        gate1 = torch.sigmoid(self.v1_out_gate(query))  # (S, N, E)
        gate2 = torch.sigmoid(self.v2_out_gate(query))  # (S, N, E)
        
        attn_output = (attn_output1 * gate1 + attn_output2 * gate2) / 2.0

        # Final linear projection
        attn_output = self.out_proj(attn_output)
        
        # Average attention weights if needed
        if need_weights:
            if average_attn_weights:
                attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            else:
                attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            score_components = None
            if return_score_components:
                if average_attn_weights:
                    def _avg_heads(x):
                        x = x.view(bsz, self.num_heads, tgt_len, src_len)
                        return x.sum(dim=1) / self.num_heads

                    score_components = {
                        'gram': _avg_heads(attn_output_weights_gram.detach()),
                        'mod1': _avg_heads(attn_output_weights1.detach()),
                        'mod2': _avg_heads(attn_output_weights2.detach()),
                        'combined_logits': _avg_heads(attn_output_weights_logits.detach()),
                    }
                else:
                    score_components = {
                        'gram': attn_output_weights_gram.detach().view(bsz, self.num_heads, tgt_len, src_len),
                        'mod1': attn_output_weights1.detach().view(bsz, self.num_heads, tgt_len, src_len),
                        'mod2': attn_output_weights2.detach().view(bsz, self.num_heads, tgt_len, src_len),
                        'combined_logits': attn_output_weights_logits.detach().view(bsz, self.num_heads, tgt_len, src_len),
                    }
            return attn_output, attn_output_weights, score_components
        else:
            return attn_output, None, None

class TransformerEncoderLayer_GRAM(nn.Module):
    """Implements encoder layer block.
    
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """Instantiate TransformerEncoderLayer Module.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int, optional): Number of heads. Defaults to 4.
            attn_dropout (float, optional): Dropout for attention mechanism. Defaults to 0.1.
            relu_dropout (float, optional): Dropout after ReLU. Defaults to 0.1.
            res_dropout (float, optional): Dropout after residual layer. Defaults to 0.1.
            attn_mask (bool, optional): Whether to apply an attention mask or not. Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = CustomMultiheadAttention_GRAM(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # The "Add & Norm" part in the paper
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, return_attn=False):
        """
        Apply TransformerEncoderLayer to Layer Input.
        
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self._maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        attn_weights, score_components = None, None
        if x_k is None and x_v is None:
            x, attn_weights, score_components = self.self_attn(
                query=x,
                mod1=x,
                mod2=x,
                attn_mask=mask,
                need_weights=return_attn,
                average_attn_weights=not return_attn,
                return_score_components=return_attn,
            )
        else:
            x_k = self._maybe_layer_norm(0, x_k, before=True)
            x_v = self._maybe_layer_norm(0, x_v, before=True)

            x, attn_weights, score_components = self.self_attn(
                query=x,
                mod1=x_k,
                mod2=x_v,
                attn_mask=mask,
                need_weights=return_attn,
                average_attn_weights=not return_attn,
                return_score_components=return_attn,
            )
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(0, x, after=True)

        residual = x
        x = self._maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(1, x, after=True)
        if return_attn:
            return x, {
                'weights': attn_weights,
                'scores': score_components,
            }
        return x

    def _maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x   

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    """Generate buffered future mask.

    Args:
        tensor (torch.Tensor): Tensor to initialize mask from.
        tensor2 (torch.Tensor, optional): Tensor to initialize target mask from. Defaults to None.

    Returns:
        torch.Tensor: Buffered future mask.
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device(tensor.device if torch.cuda.is_available() else "cpu"))
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    """Generate Linear Layer with given parameters and Xavier initialization.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool, optional): Whether to include a bias term or not. Defaults to True.

    Returns:
        nn.Module: Initialized Linear Module.
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    """Generate LayerNorm Layer with given parameters.

    Args:
        embedding_dim (int): Embedding dimension

    Returns:
        nn.Module: Initialized LayerNorm Module
    """
    m = nn.LayerNorm(embedding_dim)
    return m


"""Implements Positional Encoding.

Adapted from fairseq repo.
"""
def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    
    Args:
        tensor (torch.Tensor): Tensor to generate padding on.   
        padding_idx (int): Position numbers start at padding_idx + 1
        left_pad (bool): Whether to pad from the left or from the right.

    Returns:
        torch.Tensor: Padded output
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(
        make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos,
                     out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[
        :tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - \
            mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0):
        """Instantiate SinusoidalPositionalEmbedding Module.

        Args:
            embedding_dim (int): Embedding dimension
            padding_idx (int, optional): Padding index. Defaults to 0.
            left_pad (int, optional): Whether to pad from the left or not. Defaults to 0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        # device --> actual weight; due to nn.DataParallel :-(
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Apply PositionalEncodings to Input.
        
        Input is expected to be of size [bsz x seqlen].

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape((bsz, seq_len, -1)).detach()

# GRAMformer
Official implementation for GRAMformer: Any-Order Modality Interactions\\via Volumetric Multimodal Cross-Attention

Example of Volumetric Multimodal Attention (VMA) for three modalities:

```python
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
```

![](att_visualization.png)


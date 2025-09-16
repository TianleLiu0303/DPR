import torch
from AAAI2025.DPR.model.dit import DiTBlock

def test_ditblock_forward_shapes():
    B = 4
    x = torch.randn(B, 32, 256)
    cross_c = torch.randn(B, 336, 256)
    y = torch.randn(B, 256)
    attn_mask = torch.zeros(B, 32, dtype=torch.bool)

    block = DiTBlock(dim=256, heads=8, dropout=0.0, mlp_ratio=4.0)
    out = block(x, cross_c, y, attn_mask)
    assert out.shape == (B, 32, 256), f"Output shape mismatch: {out.shape}"

def test_ditblock_forward_different_batch():
    for B in [1, 2, 8]:
        x = torch.randn(B, 32, 256)
        cross_c = torch.randn(B, 336, 256)
        y = torch.randn(B, 256)
        attn_mask = torch.zeros(B, 32, dtype=torch.bool)
        block = DiTBlock(dim=256, heads=4, dropout=0.0, mlp_ratio=2.0)
        out = block(x, cross_c, y, attn_mask)
        assert out.shape == (B, 32, 256)

def test_ditblock_forward_masking():
    B = 2
    x = torch.randn(B, 32, 256)
    cross_c = torch.randn(B, 336, 256)
    y = torch.randn(B, 256)
    attn_mask = torch.ones(B, 32, dtype=torch.bool)  # all masked
    block = DiTBlock(dim=256, heads=2, dropout=0.0, mlp_ratio=2.0)
    out = block(x, cross_c, y, attn_mask)
    assert out.shape == (B, 32, 256)

def test_ditblock_grad():
    B = 2
    x = torch.randn(B, 32, 256, requires_grad=True)
    cross_c = torch.randn(B, 336, 256)
    y = torch.randn(B, 256)
    attn_mask = torch.zeros(B, 32, dtype=torch.bool)
    block = DiTBlock(dim=256, heads=2, dropout=0.0, mlp_ratio=2.0)
    out = block(x, cross_c, y, attn_mask)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def main():
    test_ditblock_forward_shapes()
    test_ditblock_forward_different_batch()
    test_ditblock_forward_masking()
    test_ditblock_grad()
    print("All tests passed.")

if __name__ == "__main__":
    main()
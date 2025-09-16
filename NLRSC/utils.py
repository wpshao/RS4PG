import numpy as np
import torch
import torch.nn.functional as F

def feats_sampling(
    I_RS: torch.Tensor,
    D_corr: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros", 
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp RS image to GS frame using correction field D_corr (Eq.11 implementation)
    
    Implements: I_0^GS = I_0^RS warped by -D_corr
    
    Args:
        I_RS: Rolling shutter image tensor of shape (B, C, H, W)
        D_corr: Correction field tensor of shape (B, H, W, 2)
               Contains [Δu, Δv] displacement vectors for each pixel
        interpolation: Interpolation method for sampling
        padding_mode: Padding mode for out-of-bound coordinates
        align_corners: Whether to align corner pixels
        
    Returns:
        I_GS: Corrected global shutter image of shape (B, C, H, W)
    """
    import torch.nn.functional as F
    
    # Validate spatial dimensions match
    if I_RS.size()[-2:] != D_corr.size()[1:3]:
        raise ValueError(
            f"Spatial dimensions mismatch: RS image {I_RS.size()[-2:]} vs " 
            f"correction field {D_corr.size()[1:3]}"
        )
    
    batch_size, channels, h, w = I_RS.shape
    
    # Create canonical coordinate grid (u, v) ∈ [0, w-1] × [0, h-1]
    grid_v, grid_u = torch.meshgrid(
        torch.arange(h, device=I_RS.device, dtype=torch.float32),
        torch.arange(w, device=I_RS.device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Stack into grid tensor of shape (h, w, 2)
    canonical_grid = torch.stack([grid_u, grid_v], dim=-1)
    canonical_grid = canonical_grid.type_as(I_RS)
    canonical_grid.requires_grad = False
    
    # Apply correction: warped_grid = canonical_grid + D_corr
    # For RS→GS correction: warped_grid = [u, v] + [Δu, Δv]
    warped_grid = canonical_grid + D_corr
    
    # Normalize warped coordinates to [-1, 1] range for grid_sample
    warped_grid_u = 2.0 * warped_grid[..., 0] / max(w - 1, 1) - 1.0
    warped_grid_v = 2.0 * warped_grid[..., 1] / max(h - 1, 1) - 1.0
    normalized_grid = torch.stack([warped_grid_u, warped_grid_v], dim=-1)
    
    # Sample RS image at corrected coordinates to obtain GS image
    I_GS = F.grid_sample(
        I_RS,
        normalized_grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    
    return I_GS


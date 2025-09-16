import torch
from einops import rearrange



def quadratic_flow(F0n1: torch.Tensor, F01: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
    """NL-RSC Solver for quadratic motion model (Proposition 2)
    
    Implements Eq.(11): m_0^GS = m_0 + D_corr
    where D_corr = [t_RS→GS, 0.5*t_RS→GS^2] · [Ḋ_u, Ḋ_v; Ẅ_u, Ẅ_v]^T
    
    Args:
        F0n1: Optical flow from I_0^RS to I_{-1}^RS (F_{0→-1})
        F01: Optical flow from I_0^RS to I_1^RS (F_{0→1})  
        gamma: Readout ratio (scanning time per row)
        tau: Target timestamp (typically 0 for GS frame)
        
    Returns:
        D_corr: Correction field to rectify RS to GS frame
    """
    batch_size, h, w, _ = F0n1.shape
    
    # Compute time parameters (Eq.13 variant for each flow)
    t_minus1 = -1 + (gamma / h) * F0n1[..., 1]  # t_{-1} for flow 0→-1
    t_plus1 = 1 + (gamma / h) * F01[..., 1]     # t_1 for flow 0→1
    
    # Build linear system A · M = B to solve for motion matrix
    # A = [[t_{-1}, 0.5*t_{-1}^2], [t_1, 0.5*t_1^2]] (Eq.12 form)
    A = torch.stack([
        torch.stack([t_minus1, 0.5 * t_minus1**2], dim=-1),
        torch.stack([t_plus1, 0.5 * t_plus1**2], dim=-1)
    ], dim=-2)
    
    # B = [F_{0→-1}, F_{0→1}]^T
    B = torch.stack([F0n1, F01], dim=-2)
    
    # Solve for motion matrix M = [[Ḋ_u, Ḋ_v], [Ẅ_u, Ẅ_v]]
    M = torch.linalg.solve(A, B)
    
    # Extract velocity (Ḋ) and acceleration (Ẅ) components
    C_dot = M[..., 0, :]    # [Ḋ_u, Ḋ_v]
    C_ddot = M[..., 1, :]   # [Ẅ_u, Ẅ_v]
    
    # Create y-coordinate grid (v0 values)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=F0n1.device, dtype=torch.float32),
        torch.arange(w, device=F0n1.device, dtype=torch.float32),
        indexing='ij'
    )
    v0 = grid_y  # v0 coordinate values
    
    # Compute RS-to-GS time offset (Eq.13): t_RS→GS = - (γ/h) * v0
    t_RS_to_GS = - (gamma / h) * v0
    
    # Compute correction vector D_corr (Eq.12, 14)
    D_corr_u = t_RS_to_GS * C_dot[..., 0] + 0.5 * t_RS_to_GS**2 * C_ddot[..., 0]
    D_corr_v = t_RS_to_GS * C_dot[..., 1] + 0.5 * t_RS_to_GS**2 * C_ddot[..., 1]
    
    D_corr = torch.stack([D_corr_u, D_corr_v], dim=-1)
    
    return D_corr


import torch
import torch.nn as nn
import torch.nn.functional as F

class CAPIProjection(nn.Module):
    """
    Compact Algebraic Projection Interface (CAPI) Module.
    Projects feature vectors into the Lie Algebra of SO(m) (Skew-Symmetric Matrices).
    """
    def __init__(self, in_features, m=32):
        super().__init__()
        self.m = m
        # Projection layer: maps input features to m*m flattened items
        self.proj = nn.Linear(in_features, m * m)
        
        # Initialize with small weights to ensure we start near identity geometry
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Args:
            x: Input features (B, C) or (B, C, H, W)
        Returns:
            X: Skew-symmetric matrices (B, m, m)
        """
        # Handle spatial features if necessary (Global Average Pooling)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # Project to m*m
        flat_matrices = self.proj(x)
        
        # Reshape to (B, m, m)
        matrices = flat_matrices.view(-1, self.m, self.m)
        
        # Enforce Anti-Symmetry: X = A - A^T
        # This ensures X lies in the tangent space of SO(m) (Lie Algebra so(m))
        X = matrices - matrices.transpose(-2, -1)
        
        return X

class CAPILoss(nn.Module):
    """
    Lie-Algebra Alignment Loss.
    Constraints:
    1. Commutator Constraint: [Xi, Xj] -> 0 for same class (Abelian subgroup alignment)
    2. Geometric Separation: Distance > gamma for different classes
    """
    def __init__(self, lambda_lie=1.0, gamma=1.0):
        super().__init__()
        self.lambda_lie = lambda_lie
        self.gamma = gamma

    def forward(self, x_lie, labels):
        """
        Args:
            x_lie: Lie Algebra elements (B, m, m) from CAPIProjection
            labels: Ground truth labels (B)
        """
        batch_size = x_lie.size(0)
        
        # Create mask for same class and diff class
        # (B, 1) == (1, B) -> (B, B) matrix of booleans
        labels = labels.view(-1, 1)
        mask_same = (labels == labels.T).float()
        mask_diff = 1.0 - mask_same
        
        # Remove self-loops from same mask (diagonal)
        mask_same = mask_same - torch.eye(batch_size, device=x_lie.device)
        
        # 1. Commutator Loss for Same Class
        # [Xi, Xj] = Xi*Xj - Xj*Xi
        # We need to compute this for all pairs efficiently.
        # But constructing (B, B, m, m) might be heavy if B is large.
        # Let's iterate or use broadcasting carefully.
        # B is usually 32-64, m is ~32. B^2 is 1024-4096. 4096 matmuls is fine.
        
        # Reshape for broadcasting
        # Xi: (B, 1, m, m)
        # Xj: (1, B, m, m)
        Xi = x_lie.unsqueeze(1)
        Xj = x_lie.unsqueeze(0)
        
        # Commutator matrix Cij = XiXj - XjXi
        # Matmul broadcasts: (B, 1, m, m) @ (1, B, m, m) -> (B, B, m, m)
        XiXj = torch.matmul(Xi, Xj)
        XjXi = torch.matmul(Xj, Xi)
        Commutator = XiXj - XjXi
        
        # Norm of commutators
        # Frobenius norm squared: sum of squares of elements
        norm_comm = torch.sum(Commutator ** 2, dim=(-2, -1)) # (B, B)
        
        loss_comm = (norm_comm * mask_same).sum() / (mask_same.sum() + 1e-8)
        
        # 2. Separation Loss for Different Class
        # Dist = || Xi - Xk ||_F^2
        # Xi - Xk: (B, B, m, m)
        Diff = Xi - Xj
        norm_diff = torch.sum(Diff ** 2, dim=(-2, -1)) # (B, B)
        
        # ReLU(gamma - dist)
        loss_sep_mat = F.relu(self.gamma - norm_diff)
        loss_sep = (loss_sep_mat * mask_diff).sum() / (mask_diff.sum() + 1e-8)
        
        total_loss = loss_comm + loss_sep
        
        return total_loss * self.lambda_lie

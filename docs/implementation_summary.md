# CAPI Implementation Summary

This document summarizes the technical implementation of the **Compact Algebraic Projection Interface (CAPI)**, designed to introduce geometric awareness into fine-grained visual classification models via Lie Algebra constraints.

## 1. Core Algorithm (`src/capi.py`)

The core logic implements the mathematical mapping from Euclidean feature space to the Lie Algebra of the Special Orthogonal Group $\mathfrak{so}(m)$.

### **CAPIProjection Module**
*   **Function**: Maps input features $F$ (from backbone) to Skew-Symmetric matrices $\mathbf{X}$.
*   **Implementation**:
    1.  **Linear Projection**: A learnable linear layer maps $F \in \mathbb{R}^{d}$ to a flattened vector of size $m^2$.
    2.  **Reshape**: Reshapes vector to $m \times m$ matrices.
    3.  **Anti-Symmetry Constraint**: Computes $\mathbf{X} = \mathbf{A} - \mathbf{A}^T$.
        *   This guarantees $\mathbf{X}^T = -\mathbf{X}$, a fundamental property of elements in $\mathfrak{so}(m)$.

### **CAPILoss Module**
Implements the specific geometric regularization functions:
1.  **Commutator Constraint (Intra-class)**: 
    *   Formula: $L_{comm} = \sum \| [\mathbf{X}_i, \mathbf{X}_j] \|_F^2$ where $[\mathbf{X}_i, \mathbf{X}_j] = \mathbf{X}_i\mathbf{X}_j - \mathbf{X}_j\mathbf{X}_i$.
    *   **Goal**: Forces same-class features to commute. Geometrically, this restricts them to an Abelian subalgebra, meaning the manifold curvature between them is zero (flat), allowing linear interpolation to approximate geodesics.
2.  **Separation Constraint (Inter-class)**:
    *   Formula: $L_{sep} = \text{ReLU}(\gamma - \| \mathbf{X}_i - \mathbf{X}_k \|_F^2)$.
    *   **Goal**: Ensures distinct classes are separated by a margin $\gamma$ in the tangent space.

---

## 2. Model Architecture (`src/model.py`)

The architecture follows a dual-branch, model-agnostic design.

*   **Backbone**: Powered by `timm` (PyTorch Image Models). This allows easy switching between ResNet, ViT, EfficientNet, etc., with a flag change.
*   **Branch 1 (Classification)**:
    *   Standard `nn.Linear` classifier.
    *   Optimized by Cross-Entropy Loss to ensure high classification accuracy.
*   **Branch 2 (Geometric Regularization)**:
    *   Contains the `CAPIProjection` layer.
    *   Outputs Lie Algebra elements used solely for computing `CAPILoss`.
*   **Inference**:
    *   The model returns both logits and Lie features, but for standard deployment, only the logits are needed. The backbone features are implicitly structured by the auxiliary task during training.

---

## 3. Training Pipeline (`src/main.py`)

A complete, clean training loop tailored for research stability.

*   **Integrated Data Interface**: Uses `src/ufgvc.py` for seamless handling of datasets like CUB-200-2011, NAbirds, etc., with auto-download capability.
*   **Loss Integration**:
    *   Total Loss $= L_{CE} + \lambda_{lie} \times L_{Lie}$.
*   **Logging & Monitoring**:
    *   **No TQDM**: Clean, line-by-line logging suitable for cluster/HPC output files (`> log.txt`).
    *   **Best Model Tracking**: Automatically checks validation accuracy after every epoch and prints `new best test acc` when a record is improved.
*   **Reproducibility**: Includes explicit seeding (`--seed`) for consistent comparisons.

## 4. Usage Example

```bash
# Basic usage with ResNet50 on CUB-200
python src/main.py --dataset cub_200_2011 --model resnet50 --batch_size 32 --capi_dim 32 --lambda_lie 1.0

# Using a Vision Transformer
python src/main.py --dataset nabirds --model vit_base_patch16_224 --pretrained
```

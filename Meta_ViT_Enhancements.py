import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Implementation of Novel Enhancements from "Comparative Analysis and Novel 
# Enhancements of Vision Transformer Architectures: SAM 2, DeiT, and DINOv3"
# ==============================================================================

class CrossAttention(nn.Module):
    """Simple Cross-Attention block for adapters."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ------------------------------------------------------------------------------
# Enhancement 1: DINOv3-Augmented SAM 2 Encoder
# ------------------------------------------------------------------------------
class AugmentedSAM2Encoder(nn.Module):
    """
    Integrates DINOv3's frozen dense features as additional input to SAM 2's image 
    encoder via a lightweight cross-attention adapter.
    """
    def __init__(self, dim=768):
        super().__init__()
        # Mock SAM 2 backbone layer
        self.sam2_block = nn.Linear(dim, dim) 
        
        # Novel cross-attention adapter to fuse DINOv3 features
        self.dino_adapter = CrossAttention(dim=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, sam2_features, frozen_dino_features):
        """
        sam2_features: [Batch, N, Dim] - Activations from SAM 2 backbone
        frozen_dino_features: [Batch, M, Dim] - Frozen outputs from DINOv3
        """
        # Process SAM 2 features
        x = self.sam2_block(sam2_features)
        
        # Enhancement 1: F_aug = F_img + CrossAttn(F_img, F_DINO)
        x_norm = self.norm1(x)
        dino_norm = self.norm2(frozen_dino_features)
        
        augmented_features = x + self.dino_adapter(x_norm, dino_norm)
        return augmented_features


# ------------------------------------------------------------------------------
# Enhancement 2: Dynamic Register + Gram Distillation for DeiT
# ------------------------------------------------------------------------------
class GramDistillationLoss(nn.Module):
    """
    Extends DeiT's distillation token with DINOv3-style registers and Gram 
    anchoring during supervised training.
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_gram_matrix(self, x):
        """Computes normalized Gram matrix for dense features."""
        B, N, C = x.shape
        # Normalize features
        x_norm = F.normalize(x, p=2, dim=-1)
        # Gram matrix: X * X^T
        gram = torch.bmm(x_norm, x_norm.transpose(1, 2)) 
        return gram

    def forward(self, student_logits, teacher_logits, student_patches, teacher_patches, labels):
        """
        student_logits: Predictions from distillation token
        teacher_logits: Predictions from Teacher model
        student_patches: Dense patch features + registers [B, N, C]
        teacher_patches: Frozen teacher patch features [B, N, C]
        """
        # 1. Hard Distillation Loss (from DeiT)
        hard_labels = teacher_logits.argmax(dim=-1)
        distill_loss = self.ce_loss(student_logits, hard_labels)
        
        # 2. Standard Supervised Loss (from Class Token)
        cls_loss = self.ce_loss(student_logits, labels)
        
        # 3. Novel Gram Anchoring Loss (Inspired by DINOv3 applied to DeiT)
        student_gram = self.compute_gram_matrix(student_patches)
        with torch.no_grad():
            teacher_gram = self.compute_gram_matrix(teacher_patches)
            
        gram_loss = F.mse_loss(student_gram, teacher_gram)
        
        # Total Novel Objective
        total_loss = 0.5 * cls_loss + 0.5 * distill_loss + 0.1 * gram_loss
        return total_loss


# ------------------------------------------------------------------------------
# Enhancement 3: Prompt-Conditioned Self-Supervised Objective
# ------------------------------------------------------------------------------
class PromptConditionedDINO(nn.Module):
    """
    Unifies prompt encoding (from SAM 2) into DINOv3's teacher-student loop.
    """
    def __init__(self, dim=768):
        super().__init__()
        # Mock projectors for global representation
        self.student_proj = nn.Linear(dim, dim)
        self.teacher_proj = nn.Linear(dim, dim)
        
        # Cross-attention to condition the student on prompts
        self.prompt_conditioner = CrossAttention(dim=dim)

    def forward(self, student_features, teacher_features, synthetic_prompts):
        """
        student_features: [B, N, C]
        teacher_features: [B, N, C] - from EMA teacher
        synthetic_prompts: [B, P, C] - Generated point/box embeddings
        """
        # Condition student on prompts (SAM 2 paradigm)
        conditioned_student = student_features + self.prompt_conditioner(student_features, synthetic_prompts)
        
        # Project to DINO objective space
        s_proj = self.student_proj(conditioned_student.mean(dim=1))
        with torch.no_grad():
            t_proj = self.teacher_proj(teacher_features.mean(dim=1))
            
        # Novel Loss: L_prompt-DINO = L_DINO + \lambda * L_prompt-cross
        # Here we approximate the DINO cross-entropy between student and teacher
        s_probs = F.log_softmax(s_proj, dim=-1)
        t_probs = F.softmax(t_proj, dim=-1)
        
        dino_loss = -(t_probs * s_probs).sum(dim=-1).mean()
        
        return dino_loss

# Example Usage Test
if __name__ == "__main__":
    print("Initializing Meta ViT Enhancements Validation...")
    
    B, N, M, C = 2, 196, 50, 768
    
    # Test 1
    sam2_feats = torch.randn(B, N, C)
    dino_feats = torch.randn(B, M, C)
    model1 = AugmentedSAM2Encoder(dim=C)
    out1 = model1(sam2_feats, dino_feats)
    print(f"Enhancement 1 output shape: {out1.shape} (Expected: {sam2_feats.shape})")
    
    # Test 2
    loss_fn = GramDistillationLoss()
    s_logits, t_logits = torch.randn(B, 1000), torch.randn(B, 1000)
    labels = torch.randint(0, 1000, (B,))
    # For distillation, teacher and student process the same image, hence same N
    teacher_feats_for_distill = torch.randn(B, N, C) 
    loss2 = loss_fn(s_logits, t_logits, sam2_feats, teacher_feats_for_distill, labels)
    print(f"Enhancement 2 Gram Distillation Loss: {loss2.item():.4f}")
    
    # Test 3
    prompts = torch.randn(B, 5, C)
    model3 = PromptConditionedDINO(dim=C)
    loss3 = model3(sam2_feats, dino_feats, prompts)
    print(f"Enhancement 3 Prompt-Conditioned Loss: {loss3.item():.4f}")

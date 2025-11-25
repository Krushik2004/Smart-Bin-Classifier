import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
class CLIPQuantityMatcher(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        quantity_dim: int = 32,
        hidden_dim: int = 256,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip.eval()
        self.config = self.clip.config
        
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
        
        embed_dim = self.config.projection_dim  # CLIP projection dim (e.g. 512)
        
        self.quantity_mlp = nn.Sequential(
            nn.Linear(1, quantity_dim),
            nn.ReLU(),
            nn.Linear(quantity_dim, quantity_dim),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + quantity_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)  # binary logit
        )
    
    def forward(self, pixel_values, input_ids, attention_mask, quantities):
        # CLIP forward
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_embeds = outputs.image_embeds   # (B, D)
        text_embeds  = outputs.text_embeds    # (B, D)
        
        # quantities: shape (B,) → (B,1)
        q = quantities.unsqueeze(-1)  # normalize if you want, e.g. q/10
        q_emb = self.quantity_mlp(q)  # (B, quantity_dim)
        
        x = torch.cat([image_embeds, text_embeds, q_emb], dim=-1)
        logits = self.classifier(x).squeeze(-1)  # (B,)
        return logits

# class CLIPQuantityMatcher(nn.Module):
#     def __init__(
#         self,
#         clip_model_name: str = "openai/clip-vit-base-patch32",
#         quantity_dim: int = 32,
#         hidden_dim: int = 256,
#         freeze_clip: bool = True,
#     ):
#         super().__init__()
#         self.clip = CLIPModel.from_pretrained(clip_model_name)
#         self.clip.eval()
#         self.config = self.clip.config
        
#         if freeze_clip:
#             for p in self.clip.parameters():
#                 p.requires_grad = False
        
#         embed_dim = self.config.projection_dim  # usually 512
        
#         # Quantity projection to embed_dim
#         self.quantity_mlp = nn.Sequential(
#             nn.Linear(1, quantity_dim),
#             nn.ReLU(),
#             nn.Linear(quantity_dim, embed_dim),
#             nn.ReLU(),
#         )

#         # Learnable CLS token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

#         # Pre-norms
#         self.ln1 = nn.LayerNorm(embed_dim)
#         # self.ln2 = nn.LayerNorm(embed_dim)2

#         # Single-head attention
#         self.attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=1,
#             batch_first=True
#         )

#         # # MLP block (Transformer feedforward)
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(embed_dim, 4 * embed_dim),
#         #     nn.GELU(),
#         #     nn.Linear(4 * embed_dim, embed_dim),
#         # )

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
    
#     def forward(self, pixel_values, input_ids, attention_mask, quantities):
#         outputs = self.clip(
#             pixel_values=pixel_values,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         image_embeds = outputs.image_embeds   # (B, D)
#         text_embeds  = outputs.text_embeds    # (B, D)

#         # Quantity → (B, D)
#         q = quantities.unsqueeze(-1)
#         q_emb = self.quantity_mlp(q)

#         B = image_embeds.size(0)

#         # CLS token
#         cls = self.cls_token.expand(B, 1, -1)  # (B,1,D)

#         # Sequence: [CLS, image, text, quantity]
#         seq = torch.cat([
#             cls,
#             image_embeds.unsqueeze(1),
#             text_embeds.unsqueeze(1),
#             q_emb.unsqueeze(1)
#         ], dim=1)  # (B,4,D)

#         # --- Transformer block ---

#         # 1. Pre-norm attention
#         seq_norm = self.ln1(seq)
#         attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm)
#         # seq = seq + attn_out  # residual

#         # # 2. Pre-norm MLP
#         # seq_norm = self.ln2(seq)
#         # seq = seq + self.mlp(seq_norm)

#         # Output = CLS token
#         cls_out = attn_out[:, 0]

#         logits = self.classifier(cls_out).squeeze(-1)
#         return logits


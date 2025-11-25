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



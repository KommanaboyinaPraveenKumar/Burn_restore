"""
model.py
========
Refactored Multimodal Forensic Restorer (ForensicT5Restorer).

Architecture:
  - Stream A (Global Context & Prompt Engine): Pretrained T5 encoder-decoder
    ingesting full-page document context paired with the surviving line prefix.
  - Stream B (Local Forensic Eye): High-resolution ResNet34 processing local
    128x256 line burn crops, extracting sub-character ghost strokes.
  - Cross-Modal Fusion: Spatial visual tokens projected to T5 embedding space
    and prepended as a visual prefix into the T5 encoder.
  - Grounded Seq2Seq Generation: T5 autoregressive decoder simultaneously attends
    to linguistic semantics and forensic visual ghost strokes to rank and emit
    the exact severed line ending.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerBase

from config import Config


class ForensicT5Restorer(nn.Module):
    """Line-Grounded Cross-Modal Document Restorer coupling full-page linguistic
    context with high-resolution ghost stroke visual tokens via T5 Seq2Seq."""

    def __init__(
        self,
        config: Config,
        vocab_size: Optional[int] = None,
        pad_id: Optional[int] = None,
        mask_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.text_model_name)
        d_model = self.t5.config.d_model  # 512 for t5-small

        # High-Resolution Vision Backbone (Local Forensic Eye)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if config.pretrained_vision else None)
        # Drop the final pooling and classification layer to retain spatial features
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet34 outputs (B, 512, H/32, W/32). For 128x256 input -> (B, 512, 4, 8) = 32 spatial tokens
        self.visual_proj = nn.Linear(512, d_model)
        self.visual_norm = nn.LayerNorm(d_model)

        self.pad_id = self.t5.config.pad_token_id

    def encode_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_burn_crop: torch.Tensor,
    ) -> Tuple[object, torch.Tensor]:
        """Extracts spatial visual tokens from the line crop and prepends them
        to the text prompt tokens in the T5 multimodal encoder."""
        b = input_ids.size(0)

        # 1. Extract visual ghost stroke tokens from the local line crop
        fmap = self.visual_backbone(line_burn_crop)  # (B, 512, 4, 8)
        vis_tokens = fmap.flatten(2).transpose(1, 2)  # (B, 32, 512)
        vis_tokens = self.visual_norm(self.visual_proj(vis_tokens))  # (B, 32, d_model)

        # 2. Extract text prompt embeddings from T5
        text_embeds = self.t5.shared(input_ids)  # (B, L, d_model)

        # 3. Concatenate visual tokens + text prompt tokens
        multimodal_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
        vis_mask = torch.ones(
            (b, vis_tokens.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 4. Pass unified representations through T5 Encoder
        encoder_outputs = self.t5.encoder(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_mask,
            return_dict=True,
        )
        return encoder_outputs, multimodal_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        line_burn_crop: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # Backward compatibility arguments
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        burn_crop_image: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids if input_ids is not None else context_input_ids
        mask = attention_mask if attention_mask is not None else context_attention_mask
        crop = line_burn_crop if line_burn_crop is not None else burn_crop_image

        if labels is None and target_input_ids is not None:
            labels = target_input_ids.clone()
            labels[labels == self.pad_id] = -100

        encoder_outputs, multimodal_mask = self.encode_multimodal(ids, mask, crop)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            labels=labels,
            return_dict=True,
        )
        # Returns (loss, logits)
        budget = torch.zeros(ids.size(0), device=ids.device)
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        line_burn_crop: Optional[torch.Tensor] = None,
        line_budget: Optional[torch.Tensor] = None,
        bos_id: Optional[int] = None,
        max_len: Optional[int] = None,
        # Backward compatibility arguments
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        burn_crop_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressively decodes restored line text conditioned on both
        linguistic context and high-resolution ghost stroke visual tokens,
        strictly bounded by the physical missing width character budget."""
        self.eval()
        ids = input_ids if input_ids is not None else context_input_ids
        mask = attention_mask if attention_mask is not None else context_attention_mask
        crop = line_burn_crop if line_burn_crop is not None else burn_crop_image
        max_len = max_len or self.config.max_target_len

        encoder_outputs, multimodal_mask = self.encode_multimodal(ids, mask, crop)
        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            max_new_tokens=max_len,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        return generated

    def generate_topk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_burn_crop: torch.Tensor,
        top_k: int = 5,
        max_len: Optional[int] = None,
    ) -> List[List[torch.Tensor]]:
        """Generates the top-K beam candidates for each sample in the batch (Ithaca / DeepMind paradigm)."""
        self.eval()
        max_len = max_len or self.config.max_target_len
        bs = input_ids.shape[0]
        encoder_outputs, multimodal_mask = self.encode_multimodal(input_ids, attention_mask, line_burn_crop)
        
        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            max_new_tokens=max_len,
            num_beams=top_k,
            num_return_sequences=top_k,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        
        # Reshape generated into [batch_size, top_k, seq_len]
        batch_candidates = []
        for b in range(bs):
            sample_candidates = [generated[b * top_k + k] for k in range(top_k)]
            batch_candidates.append(sample_candidates)
        return batch_candidates


# Backward-compatible alias
GroundedDocumentRestorer = ForensicT5Restorer


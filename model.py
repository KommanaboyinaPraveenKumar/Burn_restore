"""
model.py
========
Refactored Multimodal Forensic Restorer (ForensicT5Restorer).

Architecture:
  - Stream A (Global Context & Prompt Engine): Pretrained T5 encoder-decoder
    ingesting full-page document context paired with the surviving line prefix.
  - Stream B (Local Forensic Eye): High-resolution ResNet34 with multi-scale
    feature fusion (layer3 fine strokes + layer4 semantic layout) and 2D spatial
    positional embeddings.
  - Cross-Modal Fusion: Gated Multi-Head Cross-Attention where textual hypotheses
    directly attend to visual ghost stroke tokens with a residual gate,
    counteracting visual modality neglect.
  - Contrastive Stroke Aligner: InfoNCE alignment loss between visual boundary
    ghost stroke features and target text representations.
  - Physical Grounding: Autoregressive PhysicalBudgetLogitsProcessor dynamically
    constraining T5 token generation within the geometric character budget.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_burn_dir = Path(__file__).resolve().parent
if str(_burn_dir) not in sys.path:
    sys.path.insert(0, str(_burn_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerBase, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

try:
    from config import Config
except ImportError:
    from Burn_restore.config import Config


# --------------------------------------------------------------------------- #
# Defensive ResNet-34 Architecture (Native PyTorch Fallback)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class NativeResNet34(nn.Module):
    """Native PyTorch ResNet-34 implementation matching torchvision architecture.
    Provides full multi-scale feature tap points without external wheel dependencies."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns intermediate layer3 features (fine strokes) and layer4 features (layout)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat3 = self.layer3(x)  # (B, 256, H/16, W/16)
        feat4 = self.layer4(feat3)  # (B, 512, H/32, W/32)
        return feat3, feat4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat4 = self.forward_multiscale(x)
        return feat4


def build_resnet34(pretrained: bool = True) -> nn.Module:
    """Builds ResNet-34 backbone with torchvision priority and defensive native fallback."""
    try:
        import importlib
        tv_models = importlib.import_module("torchvision.models")
        weights = getattr(tv_models, "ResNet34_Weights").DEFAULT if pretrained else None
        resnet = tv_models.resnet34(weights=weights)
        return resnet
    except Exception:
        # Fallback to pure PyTorch native implementation
        resnet = NativeResNet34(in_channels=3)
        return resnet


# --------------------------------------------------------------------------- #
# Forensic Multi-Scale Visual Encoder with 2D Spatial Positional Embeddings
# --------------------------------------------------------------------------- #
class MultiScaleForensicEncoder(nn.Module):
    """Extracts fine-grained ghost stroke features and page margin context from
    line burn crops, enriched with 2D spatial coordinates."""

    def __init__(self, d_model: int = 512, pretrained: bool = True, use_multi_scale: bool = True,
                 grid: str = "coarse"):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        # "coarse": 4x8 = 32 tokens of 32 px (legacy).  "fine": 8x16 = 128 tokens
        # of 16 px (~1 character each) - layer4 is upsampled onto the layer3 grid.
        self.grid = grid
        self.last_h2 = None  # stride-8 features, read by the CTC reading head
        backbone = build_resnet34(pretrained=pretrained)

        if hasattr(backbone, "layer1") and hasattr(backbone, "layer4"):
            # Cleanly access layers
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        else:
            # Fallback wrapper
            children = list(backbone.children())
            self.stem = nn.Sequential(*children[:4])
            self.layer1 = children[4]
            self.layer2 = children[5]
            self.layer3 = children[6]
            self.layer4 = children[7]

        # Multi-scale fusion: downsamples layer3 (256, 8, 16) to (256, 4, 8) and fuses with layer4 (512, 4, 8)
        if grid == "fine":
            self.downsample_layer3 = None
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        elif use_multi_scale:
            self.downsample_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        else:
            self.downsample_layer3 = None
            self.fusion_conv = None

        self.proj = nn.Linear(512, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 2D spatial positional embeddings: (4 x 8) coarse or (8 x 16) fine
        self.num_rows, self.num_cols = (8, 16) if grid == "fine" else (4, 8)
        self.row_embed = nn.Embedding(self.num_rows, d_model)
        self.col_embed = nn.Embedding(self.num_cols, d_model)

    @property
    def resnet_parameters(self) -> List[nn.Parameter]:
        """Returns parameters belonging to the ResNet convolutional layers (stem, layer1-4)."""
        return (
            list(self.stem.parameters())
            + list(self.layer1.parameters())
            + list(self.layer2.parameters())
            + list(self.layer3.parameters())
            + list(self.layer4.parameters())
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 128, 256) line crop image.
        Returns:
            vis_tokens: (B, 32, d_model) spatial visual tokens.
            fmap4: (B, 512, 4, 8) spatial feature map for contrastive aligner.
        """
        b = x.size(0)
        h = self.stem(x)
        h1 = self.layer1(h)
        h2 = self.layer2(h1)  # (B, 128, 16, 32)
        self.last_h2 = h2
        h3 = self.layer3(h2)  # (B, 256, 8, 16)
        h4 = self.layer4(h3)  # (B, 512, 4, 8)

        if self.grid == "fine":
            h4_up = F.interpolate(h4, size=h3.shape[-2:], mode="bilinear", align_corners=False)
            fmap = self.fusion_conv(torch.cat([h4_up, h3], dim=1))  # (B, 512, 8, 16)
        elif self.use_multi_scale and self.downsample_layer3 is not None:
            h3_down = self.downsample_layer3(h3)  # (B, 256, 4, 8)
            fused = torch.cat([h4, h3_down], dim=1)  # (B, 768, 4, 8)
            fmap = self.fusion_conv(fused)  # (B, 512, 4, 8)
        else:
            fmap = h4

        # Flatten spatial dimensions: (B, 512, 4, 8) -> (B, 32, 512)
        tokens = fmap.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)  # (B, 32, d_model)

        # Apply 2D spatial coordinate embeddings
        row_ids = torch.arange(self.num_rows, device=x.device).repeat_interleave(self.num_cols)
        col_ids = torch.arange(self.num_cols, device=x.device).repeat(self.num_rows)
        spatial_pos = self.row_embed(row_ids) + self.col_embed(col_ids)  # (32, d_model)
        tokens = self.norm(tokens + spatial_pos.unsqueeze(0))

        return tokens, fmap


# --------------------------------------------------------------------------- #
# CTC reading head (direct supervision for reading visible strokes)
# --------------------------------------------------------------------------- #
CTC_CHARSET = [chr(c) for c in range(32, 127)]          # printable ASCII; index 0 = blank
CTC_INDEX = {ch: i + 1 for i, ch in enumerate(CTC_CHARSET)}


def encode_ctc_text(text: str) -> List[int]:
    return [CTC_INDEX[ch] for ch in text if ch in CTC_INDEX]


def decode_ctc_greedy(log_probs: torch.Tensor) -> List[str]:
    """log_probs: (B, T, C) -> collapsed strings."""
    out = []
    for seq in log_probs.argmax(-1).tolist():
        chars, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                chars.append(CTC_CHARSET[k - 1])
            prev = k
        out.append("".join(chars))
    return out


class CutOffsetRegressor(nn.Module):
    """Predicts where the legible text ends: the distance in px (at 1024-px page
    scale) between the burn boundary b_x and the 50%-opacity cut, from the crop.
    The fade width varies between burns, so a fixed offset (GHOST_CUT_PX) makes
    the character budget noisy.  Small and model-independent, so every system
    (text-only included) gets the same budget."""

    def __init__(self, width: int = 32):
        super().__init__()
        c = [3, width, 2 * width, 4 * width, 4 * width]
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)          # (B, 4w, H/16, W/16) = (B, 128, 8, 16)
        self.head = nn.Sequential(nn.Linear(4 * width * 16, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.register_buffer("scale", torch.tensor(30.0))  # predict offset / 30 for a well-scaled target

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        f = self.features(crop).mean(2)                  # average over height, keep horizontal position
        return self.head(f.flatten(1)).squeeze(1) * self.scale


class CTCReadingHead(nn.Module):
    """Reads the characters visible in the crop (surviving prefix inside the crop
    plus the faint ghost characters) from stride-8 ResNet features.  Trained with
    CTC; gives the visual encoder an explicit reading signal and a
    vision-only reading measurement."""

    def __init__(self, in_ch: int = 128, hidden: int = 192, n_cols: int = 64):
        super().__init__()
        self.n_cols = n_cols
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(256, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, len(CTC_CHARSET) + 1)

    def forward(self, h2: torch.Tensor) -> torch.Tensor:
        x = self.conv(h2).mean(dim=2)                                  # (B, 256, W)
        x = F.interpolate(x, size=self.n_cols, mode="linear", align_corners=False)
        x, _ = self.rnn(x.transpose(1, 2).float())                     # (B, T, 2H)
        return self.out(x).log_softmax(-1)                             # (B, T, C)


# --------------------------------------------------------------------------- #
# Gated Cross-Modal Fusion Module
# --------------------------------------------------------------------------- #
class GatedCrossModalFusion(nn.Module):
    """Bridges visual ghost stroke evidence directly into linguistic prompt tokens
    via multi-head cross-attention with a learnable sigmoid residual gate.
    
    Prevents visual modality neglect by allowing text tokens (especially incomplete
    line prefix and infill trigger) to dynamically query spatial stroke features."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5 initial weight
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_embeds: torch.Tensor,
        vis_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_embeds: (B, L_text, d_model)
            vis_tokens: (B, N_vis, d_model)
            text_mask: (B, L_text) binary attention mask
        Returns:
            fused_text_embeds: (B, L_text, d_model)
        """
        # Multi-head cross-attention: Q=text, K=vis, V=vis
        attn_out, _ = self.cross_attn(
            query=text_embeds,
            key=vis_tokens,
            value=vis_tokens,
        )
        # Learnable gated residual connection
        gate_weight = torch.sigmoid(self.gate)
        h = self.norm1(text_embeds + gate_weight * attn_out)
        h = self.norm2(h + self.mlp(h))
        return h


class ResidualGatedFusion(nn.Module):
    """Flamingo-style gated cross-attention (Alayrac et al., 2022).

    Unlike ``GatedCrossModalFusion`` (kept for legacy checkpoints), the text
    path is never normalised: pretrained T5 embeddings pass through untouched
    and visual evidence is *added* through tanh gates initialised at zero.
    At initialisation the module is the identity, so the multimodal model
    starts from pretrained T5 and can only gain from the image.
    ``embed_scale`` (RMS of T5 input embeddings) brings the attention output
    to the scale of the token embeddings it is added to.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1,
                 embed_scale: float = 1.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("embed_scale", torch.tensor(float(embed_scale)), persistent=False)

    def forward(
        self,
        text_embeds: torch.Tensor,
        vis_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = self.norm_kv(vis_tokens)
        attn_out, _ = self.cross_attn(query=self.norm_q(text_embeds), key=kv, value=kv)
        h = text_embeds + torch.tanh(self.attn_gate) * self.embed_scale * attn_out
        h = h + torch.tanh(self.ff_gate) * self.embed_scale * self.mlp(self.norm_ff(h))
        return h


# --------------------------------------------------------------------------- #
# Contrastive Stroke Aligner (InfoNCE Supervision)
# --------------------------------------------------------------------------- #
class ContrastiveStrokeAligner(nn.Module):
    """Enforces fine-grained alignment between visual ghost strokes at the burn
    boundary and the target character/text embeddings."""

    def __init__(self, d_model: int = 512, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.vis_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

    def forward(
        self,
        fmap: torch.Tensor,
        target_embeds: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            fmap: (B, 512, 4, 8) visual feature map. Columns 5-6 hold the legible ghost zone.
            target_embeds: (B, d_model) pooled or first-token target embedding.
            confidence: Optional (B,) alignment confidence weights.
        Returns:
            loss: scalar contrastive InfoNCE loss.
        """
        b = fmap.size(0)
        # Crop spans b_x-200 .. b_x+56 px; the legible ghost zone sits around
        # x = 160..224 px of the 256 px crop (cols 5-6 of 8, or 10-13 of 16).
        w = fmap.size(-1)
        c0, c1 = (w * 160) // 256, (w * 224) // 256
        burn_edge_feat = fmap[:, :, :, c0:c1].mean(dim=(2, 3))  # (B, 512)

        z_v = F.normalize(self.vis_proj(burn_edge_feat), dim=-1)  # (B, proj_dim)
        z_t = F.normalize(self.text_proj(target_embeds), dim=-1)   # (B, proj_dim)

        if b == 1:
            # Single sample: cosine similarity loss
            cos_sim = (z_v * z_t).sum(dim=-1)
            loss = (1.0 - cos_sim).mean()
            return loss

        # Batch contrastive InfoNCE
        logits = torch.matmul(z_v, z_t.T) / self.temperature  # (B, B)
        labels = torch.arange(b, device=fmap.device)

        loss_v2t = F.cross_entropy(logits, labels, reduction="none")
        loss_t2v = F.cross_entropy(logits.T, labels, reduction="none")
        loss = 0.5 * (loss_v2t + loss_t2v)

        if confidence is not None:
            if not isinstance(confidence, torch.Tensor):
                conf = torch.tensor(confidence, device=fmap.device, dtype=torch.float32)
            else:
                conf = confidence.to(device=fmap.device, dtype=torch.float32)
            conf = conf.view(-1).clamp(min=0.1, max=1.0)
            loss = (loss * conf).sum() / conf.sum()
        else:
            loss = loss.mean()

        return loss


# --------------------------------------------------------------------------- #
# Autoregressive Physical Budget Logits Processor
# --------------------------------------------------------------------------- #
class PhysicalBudgetLogitsProcessor(LogitsProcessor):
    """Custom Hugging Face LogitsProcessor enforcing physical character budgets
    during autoregressive beam search and greedy decoding.
    
    Subword-aware: Incremental character counting decodes generated SentencePiece
    tokens, strictly excluding surviving input prompts and special tokens.
    
    Escalating EOS Bias:
      When generated characters exceed the physical missing width budget:
        bias = max(0, chars - (budget + margin_buffer)) * penalty_weight
        scores[eos_id] += bias
        scores[non_eos] -= bias
        If excess >= hard_cutoff: non-eos logits set to -inf.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        budgets: torch.Tensor,
        eos_token_id: int,
        penalty_weight: float = 4.0,
        margin_buffer: float = 2.0,
        hard_cutoff_excess: float = 5.0,
        mode: str = "soft",
    ):
        """mode "soft": escalating EOS bias once the text exceeds budget + margin (legacy).
        mode "hard": ``budgets`` are *upper bounds*; any token whose characters would
        push the decoded text past the bound is masked (look-ahead), so outputs never
        exceed it.  EOS and zero-width special tokens stay allowed."""
        super().__init__()
        self.mode = mode
        self.tokenizer = tokenizer
        # Store budgets as 1D float tensor on CPU or device
        if isinstance(budgets, (list, tuple)):
            self.budgets = torch.tensor(budgets, dtype=torch.float32)
        else:
            self.budgets = budgets.clone().float()
        self.eos_token_id = eos_token_id
        self.penalty_weight = penalty_weight
        self.margin_buffer = margin_buffer
        self.hard_cutoff_excess = hard_cutoff_excess
        self._tok_len = None  # (V,) characters each token adds; built lazily for the logits width
        self._lead_space = None

    def _token_lengths(self, vocab: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._tok_len is None or self._tok_len.shape[0] != vocab or self._tok_len.device != device:
            n_tok = len(self.tokenizer)
            pieces = self.tokenizer.convert_ids_to_tokens(list(range(min(vocab, n_tok))))
            special = set(self.tokenizer.all_special_ids)
            lens, lead = [], []
            for i, pc in enumerate(pieces):
                if i in special or pc is None:
                    lens.append(0); lead.append(False)
                else:
                    t = pc.replace("\u2581", " ")
                    lens.append(len(t)); lead.append(t.startswith(" "))
            # <unk> would surface as literal text; logits wider than the tokenizer
            # (T5 pads to 32128) are not real tokens: never allowed
            if self.tokenizer.unk_token_id is not None and self.tokenizer.unk_token_id < len(lens):
                lens[self.tokenizer.unk_token_id] = 10_000
            lens += [10_000] * (vocab - len(lens)); lead += [False] * (vocab - len(lead))
            self._tok_len = torch.tensor(lens, dtype=torch.float32, device=device)
            self._lead_space = torch.tensor(lead, dtype=torch.bool, device=device)
        return self._tok_len, self._lead_space

    def _hard(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_beams, vocab = scores.shape
        num_samples = self.budgets.shape[0]
        num_beams = max(1, batch_beams // num_samples)
        tok_len, lead = self._token_lengths(vocab, scores.device)
        if not hasattr(self, "_sentinels"):
            self._sentinels = set(self.tokenizer.additional_special_tokens_ids or [])
            self._first_sentinel = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        cur = torch.empty(batch_beams, device=scores.device)
        ub = torch.empty(batch_beams, device=scores.device)
        done = torch.zeros(batch_beams, dtype=torch.bool, device=scores.device)
        for b in range(batch_beams):
            # the answer is the span after <extra_id_0> up to the next sentinel (see extract_infill)
            seq = input_ids[b].tolist()
            start = seq.index(self._first_sentinel) + 1 if self._first_sentinel in seq else 0
            seg = seq[start:]
            stop = next((i for i, t in enumerate(seg) if t in self._sentinels), None)
            if stop is not None:
                done[b], seg = True, seg[:stop]
            # count characters from the pieces themselves (SentencePiece joins them verbatim);
            # decode() would drop a pending trailing space and undercount
            seg_t = torch.tensor(seg, dtype=torch.long, device=scores.device)
            lens_b = tok_len[seg_t] if len(seg) else tok_len[:0]
            lens_b = torch.where(lens_b >= 10_000, torch.zeros_like(lens_b), lens_b)
            nz = torch.nonzero(lens_b > 0)
            lead_cut = float(lead[seg_t[nz[0, 0]]]) if len(nz) else 0.0
            cur[b] = float(lens_b.sum()) - lead_cut
            ub[b] = float(self.budgets[min(b // num_beams, num_samples - 1)])
        # the first visible token's leading space is stripped from the output
        add = tok_len.unsqueeze(0) - ((cur == 0).unsqueeze(1) & lead.unsqueeze(0)).float()
        over = (cur.unsqueeze(1) + add > ub.unsqueeze(1)) & (tok_len.unsqueeze(0) > 0) & (ub.unsqueeze(1) > 0)
        over |= done.unsqueeze(1)          # span closed by a sentinel: only EOS may follow
        over[:, self.eos_token_id] = False
        return scores.masked_fill(over, float("-inf"))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.mode == "hard":
            return self._hard(input_ids, scores)
        batch_beams = input_ids.shape[0]
        num_samples = self.budgets.shape[0]
        num_beams = max(1, batch_beams // num_samples)

        for beam_idx in range(batch_beams):
            sample_idx = min(beam_idx // num_beams, num_samples - 1)
            budget = float(self.budgets[sample_idx].item())

            # Skip budget penalty if budget is non-positive or unknown
            if budget <= 0:
                continue

            # Decode generated tokens for this beam sequence
            tokens = input_ids[beam_idx]
            # Strip decoder start token or special tokens
            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            # Clean infilling artifacts if present in decoded stream
            if "| prefix" in decoded_text:
                decoded_text = decoded_text.replace("| prefix", "")
            current_chars = len(decoded_text)

            excess = current_chars - (budget + self.margin_buffer)
            if excess > 0:
                # Escalating additive bias
                bias = float(excess * self.penalty_weight)
                scores[beam_idx, :] = scores[beam_idx, :] - bias
                scores[beam_idx, self.eos_token_id] = scores[beam_idx, self.eos_token_id] + (2.0 * bias)

                # Hard cutoff enforcement if runaway length occurs
                if excess >= self.hard_cutoff_excess:
                    scores[beam_idx, :] = -1e4
                    scores[beam_idx, self.eos_token_id] = 100.0

        return scores


# --------------------------------------------------------------------------- #
# Main Multimodal Restorer Architecture
# --------------------------------------------------------------------------- #
class ForensicT5Restorer(nn.Module):
    """Line-Grounded Multimodal Document Restorer coupling full-page linguistic
    context with high-resolution ghost stroke visual tokens and physical grounding."""

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
        if getattr(config, "gradient_checkpointing", False):
            # Recompute T5 activations in the backward pass (t5-base on a 6 GB GPU)
            self.t5.gradient_checkpointing_enable()
            self.t5.config.use_cache = False
        d_model = self.t5.config.d_model  # 512 for t5-small

        # High-Resolution Vision Backbone with 2D Spatial Positional Embeddings
        self.visual_encoder = MultiScaleForensicEncoder(
            d_model=d_model,
            pretrained=config.pretrained_vision,
            use_multi_scale=getattr(config, "use_multi_scale_vision", True),
            grid=getattr(config, "vision_grid", "fine"),
        )
        self.reading_head = CTCReadingHead() if getattr(config, "use_reading_head", True) else None

        # Backward-compatibility alias for optimizer / fine-tuning inspection
        self.visual_backbone = self.visual_encoder
        self.visual_proj = self.visual_encoder.proj
        self.visual_norm = self.visual_encoder.norm

        # Cross-modal fusion.  "flamingo" (default) = residual, zero-init gates,
        # text path untouched.  "legacy" = original LayerNorm fusion, kept only
        # so checkpoints trained before the fix still load.
        self.fusion_style = getattr(config, "fusion_style", "flamingo")
        with torch.no_grad():
            embed_scale = float(self.t5.shared.weight.float().pow(2).mean().sqrt())
        self.embed_scale = embed_scale
        if self.fusion_style == "legacy":
            self.cross_modal_fusion = GatedCrossModalFusion(
                d_model=d_model,
                num_heads=getattr(config, "fusion_heads", 8),
                dropout=getattr(config, "fusion_dropout", 0.1),
            )
        else:
            self.cross_modal_fusion = ResidualGatedFusion(
                d_model=d_model,
                num_heads=getattr(config, "fusion_heads", 8),
                dropout=getattr(config, "fusion_dropout", 0.1),
                embed_scale=embed_scale,
            )

        # Contrastive Stroke Aligner
        self.contrastive_aligner = ContrastiveStrokeAligner(
            d_model=d_model,
            proj_dim=128,
            temperature=getattr(config, "contrastive_temp", 0.07),
        )

        self.pad_id = self.t5.config.pad_token_id
        self.eos_id = self.t5.config.eos_token_id
        self.tokenizer = None  # Loaded lazily or passed

    def _get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name)
        return self.tokenizer

    def encode_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_burn_crop: torch.Tensor,
    ) -> Tuple[object, torch.Tensor, torch.Tensor]:
        """Extracts spatial visual tokens from line crop, fuses them with text prompt
        embeddings via gated cross-attention, and returns unified encoder representations."""
        b = input_ids.size(0)

        # Text-only ablation: the vision stream is bypassed entirely
        if not getattr(self.config, "use_vision", True):
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            return encoder_outputs, attention_mask, None

        # 1. Extract visual ghost stroke tokens with 2D spatial embeddings
        vis_tokens, fmap = self.visual_encoder(line_burn_crop)  # (B, 32, d_model), (B, 512, 4, 8)

        # 2. Extract text prompt embeddings from T5
        text_embeds = self.t5.shared(input_ids)  # (B, L, d_model)

        # 3. Gated Cross-Modal Fusion: text tokens directly query visual features
        if getattr(self.config, "use_cross_modal_fusion", True):
            text_embeds = self.cross_modal_fusion(text_embeds, vis_tokens, attention_mask)

        # 4. Concatenate visual prefix + cross-fused text tokens.  With the
        #    flamingo fusion the LayerNorm-ed (RMS ~1) visual tokens are rescaled
        #    to the RMS of T5 input embeddings so they are not drowned out.
        vis_prefix = vis_tokens if self.fusion_style == "legacy" else vis_tokens * self.embed_scale
        multimodal_embeds = torch.cat([vis_prefix, text_embeds], dim=1)
        vis_mask = torch.ones(
            (b, vis_tokens.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
        )
        multimodal_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 5. Pass through T5 Multimodal Encoder
        encoder_outputs = self.t5.encoder(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_mask,
            return_dict=True,
        )
        return encoder_outputs, multimodal_mask, fmap

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        line_burn_crop: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        alignment_confidence: Optional[torch.Tensor] = None,
        lambda_contrastive: Optional[float] = None,
        visible_ids: Optional[torch.Tensor] = None,
        ctc_targets: Optional[torch.Tensor] = None,
        ctc_lengths: Optional[torch.Tensor] = None,
        lambda_read: Optional[float] = None,
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

        encoder_outputs, multimodal_mask, fmap = self.encode_multimodal(ids, mask, crop)
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            labels=labels,
            return_dict=True,
        )

        gen_loss = outputs.loss

        # Optional Contrastive Stroke Alignment Loss
        lambda_c = (
            lambda_contrastive
            if lambda_contrastive is not None
            else kwargs.get("lambda_contrastive", getattr(self.config, "lambda_contrastive", 0.1))
        )
        # Contrastive stroke alignment is applied ONLY to target characters that
        # still have visible strokes in the crop (``visible_ids``).  Aligning the
        # image with fully-burnt text would train the encoder to hallucinate.
        contrast_loss = None
        if lambda_c > 0.0 and fmap is not None and visible_ids is not None:
            has_vis = (visible_ids != self.pad_id).any(dim=1)
            if int(has_vis.sum()) >= 2:
                v_ids = visible_ids[has_vis]
                tok_mask = (v_ids != self.pad_id).unsqueeze(-1).float()
                emb = self.t5.shared(v_ids).float()
                target_embeds = (emb * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1.0)
                conf = alignment_confidence[has_vis] if isinstance(alignment_confidence, torch.Tensor) else None
                contrast_loss = self.contrastive_aligner(fmap[has_vis], target_embeds, conf)
        total_loss = gen_loss + lambda_c * contrast_loss if contrast_loss is not None else gen_loss

        # CTC reading loss on the text visible in the crop
        read_loss = None
        lam_r = lambda_read if lambda_read is not None else getattr(self.config, "lambda_read", 0.5)
        if (self.reading_head is not None and lam_r > 0 and fmap is not None
                and ctc_targets is not None and ctc_lengths is not None and int(ctc_lengths.sum()) > 0):
            lp = self.reading_head(self.visual_encoder.last_h2)          # (B, T, C)
            T = lp.size(1)
            keep = (ctc_lengths > 0) & (ctc_lengths <= T // 2)
            if keep.any():
                read_loss = F.ctc_loss(
                    lp[keep].transpose(0, 1), ctc_targets[keep],
                    torch.full((int(keep.sum()),), T, dtype=torch.long, device=lp.device),
                    ctc_lengths[keep], blank=0, reduction="mean", zero_infinity=True,
                )
                total_loss = total_loss + lam_r * read_loss
        self.last_read_loss = float(read_loss.detach().item()) if read_loss is not None else 0.0
        self.last_gen_loss = float(gen_loss.detach().item()) if gen_loss is not None else 0.0
        self.last_contrast_loss = float(contrast_loss.detach().item()) if contrast_loss is not None else 0.0

        return total_loss, outputs.logits

    @torch.no_grad()
    def read_crop(self, line_burn_crop: torch.Tensor) -> List[str]:
        """Vision-only CTC transcription of the text visible in each crop."""
        if self.reading_head is None or not getattr(self.config, "use_vision", True):
            return [""] * line_burn_crop.size(0)
        self.visual_encoder(line_burn_crop)
        return decode_ctc_greedy(self.reading_head(self.visual_encoder.last_h2))

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        line_burn_crop: Optional[torch.Tensor] = None,
        line_budget: Optional[torch.Tensor] = None,
        bos_id: Optional[int] = None,
        max_len: Optional[int] = None,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        use_physical_grounding: Optional[bool] = None,
        budget_mode: Optional[str] = None,
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

        encoder_outputs, multimodal_mask, _ = self.encode_multimodal(ids, mask, crop)

        # Configure physical budget logits processor
        processors = LogitsProcessorList()
        apply_budget = (
            use_physical_grounding
            if use_physical_grounding is not None
            else getattr(self.config, "use_physical_budget_processor", True)
        )
        if apply_budget and line_budget is not None:
            tokenizer = self._get_tokenizer()
            budget_proc = PhysicalBudgetLogitsProcessor(
                tokenizer=tokenizer,
                budgets=line_budget,
                eos_token_id=self.eos_id,
                penalty_weight=getattr(self.config, "alignment_penalty_weight", 4.0),
                margin_buffer=getattr(self.config, "budget_margin_buffer", 2.0),
                mode=budget_mode or getattr(self.config, "budget_mode", "soft"),
            )
            processors.append(budget_proc)

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            max_new_tokens=max_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processor=processors if len(processors) > 0 else None,
        )
        return generated

    def generate_topk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_burn_crop: torch.Tensor,
        line_budget: Optional[torch.Tensor] = None,
        top_k: int = 5,
        max_len: Optional[int] = None,
        return_scores: bool = False,
        budget_mode: Optional[str] = None,
    ):
        """Generates the top-K beam candidates for each sample in the batch (Ithaca / DeepMind paradigm).
        With ``return_scores`` also returns a (B, K) tensor of length-normalised
        sequence log-probabilities (beam scores)."""
        self.eval()
        max_len = max_len or self.config.max_target_len
        bs = input_ids.shape[0]
        encoder_outputs, multimodal_mask, _ = self.encode_multimodal(input_ids, attention_mask, line_burn_crop)

        processors = LogitsProcessorList()
        if getattr(self.config, "use_physical_budget_processor", True) and line_budget is not None:
            tokenizer = self._get_tokenizer()
            budget_proc = PhysicalBudgetLogitsProcessor(
                tokenizer=tokenizer,
                budgets=line_budget,
                eos_token_id=self.eos_id,
                penalty_weight=getattr(self.config, "alignment_penalty_weight", 4.0),
                margin_buffer=getattr(self.config, "budget_margin_buffer", 2.0),
                mode=budget_mode or getattr(self.config, "budget_mode", "soft"),
            )
            processors.append(budget_proc)

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=multimodal_mask,
            max_new_tokens=max_len,
            num_beams=top_k,
            num_return_sequences=top_k,
            early_stopping=True,
            logits_processor=processors if len(processors) > 0 else None,
            return_dict_in_generate=return_scores,
            output_scores=return_scores,
        )
        scores = None
        if return_scores:
            scores = generated.sequences_scores.view(bs, top_k)
            generated = generated.sequences

        batch_candidates = []
        for b in range(bs):
            sample_candidates = [generated[b * top_k + k] for k in range(top_k)]
            batch_candidates.append(sample_candidates)
        if return_scores:
            return batch_candidates, scores
        return batch_candidates


# Backward-compatible alias
GroundedDocumentRestorer = ForensicT5Restorer


ARCH_KEYS = ("use_vision", "use_cross_modal_fusion", "use_multi_scale_vision", "text_model_name",
             "fusion_style", "vision_grid", "use_reading_head", "prompt_style")
# Value to assume when a key is missing from an older checkpoint
_LEGACY_DEFAULTS = {"fusion_style": "legacy", "vision_grid": "coarse", "use_reading_head": False,
                    "prompt_style": "v2"}


def apply_saved_architecture(cfg, saved_cfg: dict):
    """Copies architecture switches from a checkpoint ``config_dict`` onto cfg.
    Checkpoints written before ``fusion_style`` existed used the legacy fusion."""
    if not saved_cfg:
        return cfg
    for k in ARCH_KEYS:
        if k in saved_cfg:
            setattr(cfg, k, saved_cfg[k])
    for k, v in _LEGACY_DEFAULTS.items():
        if k not in saved_cfg:
            setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# Calibrated upper character budget (cut estimator)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_cut_offsets(model: "CutOffsetRegressor", crops, device, batch_size: int = 256) -> "np.ndarray":
    """Predicted b_x - cut_x (px) for an (N, H, W, 3) uint8 RGB crop array (memmap is fine)."""
    import numpy as np
    try:
        from line_data import crop_to_tensor
    except ImportError:
        from Burn_restore.line_data import crop_to_tensor
    model.eval()
    out = []
    for i in range(0, len(crops), batch_size):
        x = torch.stack([crop_to_tensor(np.asarray(c)) for c in crops[i:i + batch_size]]).to(device)
        out.append(model(x).float().cpu())
    return torch.cat(out).clamp(0, 120).numpy() if out else np.zeros(0)


def load_cut_estimator(path, device):
    """Returns (model, meta) or (None, None) if the checkpoint does not exist."""
    if not path or not os.path.isfile(path):
        return None, None
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CutOffsetRegressor().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, {"ub_ratio": float(ck["ub_ratio"]), "ub_add": float(ck["ub_add"])}


def upper_budget(estimate: float, meta: Optional[dict]) -> float:
    """Calibrated upper bound on the number of missing characters."""
    if not meta:
        return float(estimate)
    return float(meta["ub_ratio"] * estimate + meta["ub_add"])

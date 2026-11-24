"""
inference/budget_estimator.py
=============================
Physical character-budget estimation, validation and post-generation compliance.

The physical character budget represents the geometric bound on how many glyphs
can physically fit inside the fire-damaged document margin before the page edge.

This module provides:
  1. Geometric budget estimation from pixel width, font scale, and visible line density.
  2. Subword-aware token counting (filtering SentencePiece artifacts and special tokens).
  3. Pre-generation constraints and safety margins.
  4. Post-generation compliance auditing (excess characters, violation flags).
  5. Boundary-aware word snapping to preserve linguistic coherence without silent truncation.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class BudgetComplianceReport:
    """Detailed audit of generation length against the physical line budget."""
    estimated_budget: float
    budget_margin_buffer: float
    max_allowed_characters: float
    generated_character_count: int
    raw_infill_text: str
    cleaned_text: str
    excess_characters: int
    budget_violation: bool
    word_snapped_text: Optional[str] = None
    was_snapped: bool = False
    is_heuristic: bool = True
    estimation_method: str = "geometric_pixel_ratio"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_budget": round(self.estimated_budget, 2),
            "budget_margin_buffer": round(self.budget_margin_buffer, 2),
            "max_allowed_characters": round(self.max_allowed_characters, 2),
            "generated_character_count": self.generated_character_count,
            "raw_infill_text": self.raw_infill_text,
            "cleaned_text": self.cleaned_text,
            "excess_characters": self.excess_characters,
            "budget_violation": self.budget_violation,
            "word_snapped_text": self.word_snapped_text,
            "was_snapped": self.was_snapped,
            "is_heuristic": self.is_heuristic,
            "estimation_method": self.estimation_method,
        }


class BudgetEstimator:
    """Estimates and evaluates physical character capacity for damaged document lines."""

    def __init__(
        self,
        default_char_width_px: float = 14.0,
        margin_buffer: float = 2.0,
        min_char_width_px: float = 6.0,
        max_char_width_px: float = 30.0,
    ):
        self.default_char_width_px = default_char_width_px
        self.margin_buffer = margin_buffer
        self.min_char_width_px = min_char_width_px
        self.max_char_width_px = max_char_width_px

    def estimate_budget(
        self,
        damaged_width_px: int,
        intact_width_px: Optional[int] = None,
        visible_text: Optional[str] = None,
        line_height_px: Optional[int] = None,
    ) -> float:
        """Estimates missing character capacity based on damaged width and font scale.
        
        Args:
            damaged_width_px: Horizontal missing pixel span from burn front to right margin.
            intact_width_px: Optional pixel width of surviving line body.
            visible_text: Optional surviving prefix text string for dynamic font calibration.
            line_height_px: Optional line height for vertical font scale scaling.
            
        Returns:
            Non-negative float character budget.
        """
        if damaged_width_px <= 0:
            return 0.0

        char_width = self.default_char_width_px

        # Dynamic glyph width estimation from surviving visible text
        if (
            visible_text is not None
            and intact_width_px is not None
            and len(visible_text.strip()) > 3
            and intact_width_px > 50
        ):
            # Compute average pixel width per character in the surviving text
            # Ignore leading/trailing whitespace
            num_chars = len(visible_text.strip())
            est_w = float(intact_width_px) / float(num_chars)
            if self.min_char_width_px <= est_w <= self.max_char_width_px:
                char_width = est_w

        elif line_height_px is not None and line_height_px > 8:
            # Typical typography aspect ratio: glyph width is ~0.55 to 0.65 of line height
            font_scale_width = float(line_height_px) * 0.58
            if self.min_char_width_px <= font_scale_width <= self.max_char_width_px:
                char_width = font_scale_width

        raw_budget = float(damaged_width_px) / char_width
        return max(0.0, float(raw_budget))

    def evaluate_compliance(
        self,
        generated_text: str,
        budget: float,
        margin_buffer: Optional[float] = None,
        enable_word_snapping: bool = True,
    ) -> BudgetComplianceReport:
        """Audits generated restoration text against the physical budget.
        
        Args:
            generated_text: String produced by the model (e.g. infill prediction).
            budget: Physical character budget in characters.
            margin_buffer: Character allowance tolerance (defaults to self.margin_buffer).
            enable_word_snapping: Whether to compute a clean word-boundary snapped variant.
        """
        buffer = margin_buffer if margin_buffer is not None else self.margin_buffer
        max_allowed = budget + buffer

        # Clean infill text: strip special sentinel tokens, whitespace
        cleaned = self.clean_infill_text(generated_text)
        gen_len = len(cleaned)

        excess = max(0, int(math.ceil(gen_len - max_allowed)))
        is_violation = excess > 0

        snapped_text: Optional[str] = None
        was_snapped = False

        if is_violation and enable_word_snapping:
            snapped_text, was_snapped = self._snap_to_word_boundary(cleaned, int(max_allowed))

        return BudgetComplianceReport(
            estimated_budget=budget,
            budget_margin_buffer=buffer,
            max_allowed_characters=max_allowed,
            generated_character_count=gen_len,
            raw_infill_text=generated_text,
            cleaned_text=cleaned,
            excess_characters=excess,
            budget_violation=is_violation,
            word_snapped_text=snapped_text if was_snapped else cleaned,
            was_snapped=was_snapped,
            is_heuristic=True,
            estimation_method="geometric_pixel_ratio",
        )

    @staticmethod
    def clean_infill_text(raw_text: str) -> str:
        """Strips T5 sentinels (<extra_id_0>, etc.), pad tokens, and infill artifacts."""
        if not raw_text:
            return ""

        # Keep only the span between <extra_id_0> and the next sentinel / EOS;
        # anything T5 generates after <extra_id_1> is not part of the infill.
        text = raw_text
        if "<extra_id_0>" in text:
            text = text.split("<extra_id_0>", 1)[1]
        text = re.split(r"<extra_id_\d+>|</s>", text, maxsplit=1)[0]
        # Remove common special tokens
        for tok in ["<pad>", "</s>", "<s>", "<unk>", "| prefix"]:
            text = text.replace(tok, "")

        return text.strip()

    @staticmethod
    def _snap_to_word_boundary(text: str, max_chars: int) -> Tuple[str, bool]:
        """Safely snaps text to the nearest whitespace boundary within max_chars."""
        if len(text) <= max_chars:
            return text, False

        candidate = text[:max_chars]
        last_space = candidate.rfind(" ")

        # If there is a space anywhere before max_chars, snap to the complete word
        if last_space > 0:
            return candidate[:last_space].rstrip(" ,;:-."), True

        # Otherwise snap at max_chars
        return candidate.rstrip(" ,;:-."), True


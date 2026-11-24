"""
scripts/generate_paper_figures.py
=================================
Generates publication-quality figures for the research paper based on verified project data.
Saves rendered assets into paper/figures/.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
paper_fig_dir = repo_root / "paper" / "figures"
paper_fig_dir.mkdir(parents=True, exist_ok=True)

# Set clean academic styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["figure.titlesize"] = 12


def generate_baseline_chart():
    bench_p = repo_root / "docs" / "benchmark_results.json"
    with open(bench_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = data["baselines_comparison"]
    names = [
        "A: Heuristic",
        "B: Text-Only T5",
        "C: Legacy Prefix",
        "D: Spatial Vision",
        "E: Gated Fusion",
        "F: Contrastive",
        "G: Full Grounded",
    ]
    cer_vals = [baselines[k]["mean_raw_cer"] for k in baselines]
    wer_vals = [baselines[k]["mean_wer"] for k in baselines]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    rects1 = ax.bar(x - width/2, cer_vals, width, label="Mean Raw CER", color="#d97706", edgecolor="#92400e")
    rects2 = ax.bar(x + width/2, wer_vals, width, label="Mean WER", color="#2563eb", edgecolor="#1e40af")

    ax.set_ylabel("Error Metric Value (Lower is Better)")
    ax.set_title("Restoration Performance Across Baselines A through G")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out_p = paper_fig_dir / "figure2_baselines_comparison.png"
    plt.savefig(out_p)
    plt.close()
    print(f"[figure] Generated {out_p}")


def generate_ablation_chart():
    abl_p = repo_root / "docs" / "ablation_results.json"
    with open(abl_p, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    labels = [
        "ABL-01: Full Multimodal",
        "ABL-02: No Vision Encoder",
        "ABL-03: No Multi-Scale",
        "ABL-04: No Spatial Pos Embed",
        "ABL-05: No Gated Cross-Attn",
        "ABL-06: No Contrastive Loss",
        "ABL-07: No Physical Budget",
        "ABL-08: Fixed Contrast Weight",
        "ABL-09: Gen-Only Loss",
        "ABL-10: Full Reference",
    ]
    cer_vals = [abl_data[k]["mean_raw_cer"] for k in abl_data]

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=300)
    colors = ["#10b981" if "Full" in l else "#64748b" for l in labels]

    bars = ax.barh(y_pos, cer_vals, color=colors, edgecolor="#334155", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Mean Raw CER (Lower is Better)")
    ax.set_title("10-Component Ablation Study Matrix")
    ax.set_xlim(1.8, 2.5)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8)

    plt.tight_layout()
    out_p = paper_fig_dir / "figure3_ablation_study.png"
    plt.savefig(out_p)
    plt.close()
    print(f"[figure] Generated {out_p}")


def generate_burn_severity_chart():
    bench_p = repo_root / "docs" / "benchmark_results.json"
    with open(bench_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    sev = data["burn_severity_robustness"]
    profiles = ["Mild\n(10-25%)", "Moderate\n(25-50%)", "Severe\n(50-75%)", "Irregular\n(Jagged)"]
    cer_vals = [sev["mild"]["mean_raw_cer"], sev["moderate"]["mean_raw_cer"], sev["severe"]["mean_raw_cer"], sev["irregular"]["mean_raw_cer"]]
    wer_vals = [sev["mild"]["mean_wer"], sev["moderate"]["mean_wer"], sev["severe"]["mean_wer"], sev["irregular"]["mean_wer"]]

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8), dpi=300)
    color = "#b45309"
    ax1.set_xlabel("Burn Corruption Severity Profile")
    ax1.set_ylabel("Mean Raw CER", color=color)
    p1 = ax1.plot(profiles, cer_vals, marker="o", linewidth=2.2, color=color, label="Mean CER")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(1.0, 3.8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color = "#1d4ed8"
    ax2.set_ylabel("Mean WER", color=color)
    p2 = ax2.plot(profiles, wer_vals, marker="s", linewidth=2.2, linestyle="--", color=color, label="Mean WER")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(3.0, 7.5)

    plt.title("Restoration Error vs. Burn Severity Profile")
    fig.tight_layout()
    out_p = paper_fig_dir / "figure4_burn_severity.png"
    plt.savefig(out_p)
    plt.close()
    print(f"[figure] Generated {out_p}")


def generate_visual_stress_chart():
    bench_p = repo_root / "docs" / "benchmark_results.json"
    with open(bench_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data["visual_grounding_stress_tests"]
    labels = ["Normal\nForensic Crop", "Zeroed Crop\n(Zeros)", "Shuffled Pixels\n(Spatial Perm)", "Gaussian Noise\n(std=0.5)"]
    cer_vals = [tests["normal"]["mean_raw_cer"], tests["zeroed"]["mean_raw_cer"], tests["shuffled"]["mean_raw_cer"], tests["noisy"]["mean_raw_cer"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
    bars = ax.bar(labels, cer_vals, color=["#475569", "#94a3b8", "#94a3b8", "#ef4444"], width=0.5, edgecolor="#1e293b")
    ax.set_ylabel("Mean Raw CER")
    ax.set_title("Visual Modality Grounding Stress Testing")
    ax.set_ylim(2.10, 2.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_p = paper_fig_dir / "figure5_visual_stress_tests.png"
    plt.savefig(out_p)
    plt.close()
    print(f"[figure] Generated {out_p}")


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.axis("off")

    # Bounding boxes for architecture components
    # Stream A: Text Context
    bbox_text = dict(boxstyle="round,pad=0.5", facecolor="#dbeafe", edgecolor="#1e40af", linewidth=1.5)
    ax.text(0.18, 0.78, "Global Context Engine (T5-small)\nSurviving Text Prefix + Page Context\n(L tokens -> d_model=512)",
            ha="center", va="center", bbox=bbox_text, fontsize=8.5)

    # Stream B: Forensic Vision
    bbox_vis = dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#b91c1c", linewidth=1.5)
    ax.text(0.18, 0.28, "Local Forensic Eye (ResNet-34)\nSevered Line Crop (128x256 px)\nLayer3+Layer4 Fusion + 2D Spatial PE\n(32 visual tokens -> d_model=512)",
            ha="center", va="center", bbox=bbox_vis, fontsize=8.5)

    # Cross-Modal Fusion
    bbox_fusion = dict(boxstyle="round,pad=0.6", facecolor="#fef3c7", edgecolor="#b45309", linewidth=1.8)
    ax.text(0.55, 0.53, "Gated Cross-Modal Fusion\nMulti-Head Cross-Attention (h=8)\nLearnable Sigmoid Residual Gate\n[Text Queries Attend to Visual Tokens]",
            ha="center", va="center", bbox=bbox_fusion, fontsize=8.5)

    # Autoregressive Decoder + Physical Budget
    bbox_dec = dict(boxstyle="round,pad=0.6", facecolor="#d1fae5", edgecolor="#047857", linewidth=1.8)
    ax.text(0.85, 0.53, "T5 Transformer Decoder\n+\nPhysicalBudgetLogitsProcessor\nAutoregressive Infilling\nBounded by dx / w_char",
            ha="center", va="center", bbox=bbox_dec, fontsize=8.5)

    # Connecting arrows
    ax.annotate("", xy=(0.38, 0.62), xytext=(0.28, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1e40af"))
    ax.annotate("", xy=(0.38, 0.44), xytext=(0.28, 0.34), arrowprops=dict(arrowstyle="->", lw=1.5, color="#b91c1c"))
    ax.annotate("", xy=(0.72, 0.53), xytext=(0.68, 0.53), arrowprops=dict(arrowstyle="->", lw=2.0, color="#b45309"))

    ax.set_title("ForensicT5Restorer v2.0 Dual-Stream Multimodal Architecture", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    out_p = paper_fig_dir / "figure1_architecture.png"
    plt.savefig(out_p)
    plt.close()
    print(f"[figure] Generated {out_p}")


if __name__ == "__main__":
    generate_baseline_chart()
    generate_ablation_chart()
    generate_burn_severity_chart()
    generate_visual_stress_chart()
    generate_architecture_diagram()


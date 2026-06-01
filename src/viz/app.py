import inspect
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import IMG_DIR, SUBMISSION_DIR
from src.metrics import error_fn, metric_fn
import src.models.loss as loss_module
from src.models.loss import UniversalLossWrapper

STARS_PATH = Path(__file__).parent / "stars.json"


# ── Dynamic loss discovery ──────────────────────────────────────────────────────
def _discover_losses() -> dict:
    return {
        name: cls
        for name, cls in inspect.getmembers(loss_module, inspect.isclass)
        if issubclass(cls, nn.Module)
        and cls is not UniversalLossWrapper
        and cls is not nn.Module
    }

LOSS_CLASSES = _discover_losses()


# ── Stars persistence ───────────────────────────────────────────────────────────
def load_stars() -> list[dict]:
    if STARS_PATH.exists():
        return json.loads(STARS_PATH.read_text())
    return []

def save_stars(stars: list[dict]) -> None:
    STARS_PATH.write_text(json.dumps(stars, indent=2))


# ── Data helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def list_model_runs() -> list[str]:
    if not SUBMISSION_DIR.exists():
        return []
    return sorted(
        [d.name for d in SUBMISSION_DIR.iterdir() if d.is_dir() and "_submission_" in d.name],
        reverse=True,
    )

@st.cache_data
def load_split_df(run_name: str, split: str) -> pd.DataFrame | None:
    path = SUBMISSION_DIR / run_name / f"{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # test.csv stores the prediction in FaceOcclusion (submission format) — normalise
    if split == "test":
        df = df.rename(columns={"FaceOcclusion": "pred"})
        df["FaceOcclusion"] = float("nan")
        df["gender"] = float("nan")
        df["iw"] = 1.0
    return df


def apply_filters(df: pd.DataFrame, gender_sel: str, occ_min: float, occ_max: float, has_gt: bool) -> pd.DataFrame:
    value_col = "FaceOcclusion" if has_gt else "pred"
    mask = (df[value_col] >= occ_min) & (df[value_col] <= occ_max)
    if has_gt and gender_sel != "All":
        gender_val = 0.0 if gender_sel == "Female" else 1.0
        mask &= df["gender"] == gender_val
    return df[mask]


def compute_loss(loss_cls, df: pd.DataFrame) -> float | None:
    if df.empty or df["FaceOcclusion"].isna().all():
        return None
    sub = df[["pred", "FaceOcclusion", "iw"]].dropna()
    if sub.empty:
        return None
    preds = torch.tensor(sub["pred"].values, dtype=torch.float32).unsqueeze(1)
    gt    = torch.tensor(sub["FaceOcclusion"].values, dtype=torch.float32).unsqueeze(1)
    iw    = torch.tensor(sub["iw"].values, dtype=torch.float32).unsqueeze(1)
    pi    = (1 / 30 + gt)
    try:
        val = UniversalLossWrapper(loss_cls())(preds, gt, iw, pi)
        return float(val)
    except Exception:
        return None


# ── App ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Occlusion Explorer", layout="centered")
st.title("Face Occlusion Explorer")

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    runs = list_model_runs()
    if not runs:
        st.warning("No model runs found in submission/")
        st.stop()

    selected_run = st.selectbox("Model run", runs)

    split = st.radio("Split", ["train", "val", "test"])
    has_gt = split != "test"

    gender_sel = st.radio("Gender", ["All", "Female", "Male"]) if has_gt else "All"

    occ_range = st.slider("Occlusion interval (%)", 0, 100, (0, 100))
    occ_min, occ_max = occ_range[0] / 100, occ_range[1] / 100

    loss_names = list(LOSS_CLASSES.keys())
    selected_losses = st.multiselect(
        "Losses to compare",
        loss_names,
        default=loss_names[:1] if loss_names else [],
    )

# ── Load data ─────────────────────────────────────────────────────────────────────
df = load_split_df(selected_run, split)
if df is None:
    st.error(f"No {split}.csv found for run `{selected_run}`. Run the pipeline to generate split predictions.")
    st.stop()

df_filtered = apply_filters(df, gender_sel, occ_min, occ_max, has_gt)

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab_stats, tab_pic, tab_stars = st.tabs(["Statistics", "Picture", "Stars"])


# ─── Statistics ───────────────────────────────────────────────────────────────────
with tab_stats:
    st.subheader("Occlusion Distribution")

    # Histogram — GT and predictions overlaid, all available splits
    split_colors = {"train": "steelblue", "val": "darkorange", "test": "forestgreen"}
    pred_colors  = {"train": "cornflowerblue", "val": "gold", "test": "lightgreen"}
    fig, ax = plt.subplots(figsize=(7, 3))
    for s, color in split_colors.items():
        s_df = load_split_df(selected_run, s)
        if s_df is None:
            continue
        if s != "test" and "FaceOcclusion" in s_df.columns:
            ax.hist(s_df["FaceOcclusion"].dropna(), bins=60, alpha=0.5,
                    label=f"{s} GT", color=color)
        if "pred" in s_df.columns:
            ax.hist(s_df["pred"].dropna(), bins=60, alpha=0.4,
                    label=f"{s} pred", color=pred_colors[s], linestyle="dashed",
                    histtype="step", linewidth=1.5)
    ax.axvline(occ_min, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(occ_max, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"selected interval [{occ_min:.2f}, {occ_max:.2f}]")
    ax.set_xlabel("Occlusion")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Filtered rows (current split + filters): **{len(df_filtered)}** / {len(df)}")

    # Scatter plot — GT vs predicted, coloured by gender (train/val only)
    if has_gt and not df_filtered.empty and "pred" in df_filtered.columns:
        st.subheader("GT vs Predicted")
        scatter_df = df_filtered[["FaceOcclusion", "pred", "gender"]].dropna()
        if not scatter_df.empty:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            gender_palette = {0.0: ("#e07b7b", "Female"), 1.0: ("#5b8dd9", "Male")}
            for g_val, (color, label) in gender_palette.items():
                sub = scatter_df[scatter_df["gender"] == g_val]
                if not sub.empty:
                    ax2.scatter(sub["FaceOcclusion"], sub["pred"],
                                alpha=0.35, s=8, color=color, label=label)
            lims = [
                min(scatter_df["FaceOcclusion"].min(), scatter_df["pred"].min()),
                max(scatter_df["FaceOcclusion"].max(), scatter_df["pred"].max()),
            ]
            ax2.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="perfect prediction")
            ax2.set_xlabel("Ground Truth")
            ax2.set_ylabel("Predicted")
            ax2.legend(markerscale=3)
            st.pyplot(fig2)
            plt.close(fig2)

    if has_gt and not df_filtered.empty:
        st.subheader("Competition Score (filtered interval)")
        female = df_filtered[df_filtered["gender"] == 0.0][["pred", "FaceOcclusion"]].dropna()
        male   = df_filtered[df_filtered["gender"] == 1.0][["pred", "FaceOcclusion"]].dropna()
        if len(female) > 0 and len(male) > 0:
            score = metric_fn(female, male)
            err_f = error_fn(female)
            err_m = error_fn(male)
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{score:.5f}")
            c2.metric("Err Female", f"{err_f:.5f}")
            c3.metric("Err Male", f"{err_m:.5f}")
        else:
            st.info("Not enough data for both genders in the selected interval.")
    elif not has_gt:
        st.info("Competition score not available for the test split (no ground truth).")

    if has_gt and selected_losses and not df_filtered.empty:
        st.subheader("Loss Comparison (filtered interval)")
        rows = []
        for name in selected_losses:
            val = compute_loss(LOSS_CLASSES[name], df_filtered)
            rows.append({"Loss": name, "Value": f"{val:.6f}" if val is not None else "N/A"})
        st.table(pd.DataFrame(rows))
    elif not has_gt and selected_losses:
        st.info("Loss values not available for the test split (no ground truth).")


# ─── Picture ───────────────────────────────────────────────────────────────────────
with tab_pic:
    st.subheader("Random Image Viewer")

    if df_filtered.empty:
        st.warning("No images match the current filters.")
    else:
        if st.button("🎲 Load random image"):
            st.session_state["pic_idx"] = int(np.random.randint(len(df_filtered)))

        if "pic_idx" not in st.session_state:
            st.session_state["pic_idx"] = 0

        idx = st.session_state["pic_idx"] % len(df_filtered)
        row = df_filtered.iloc[idx]
        img_path = IMG_DIR / row["filename"]

        col_img, col_info = st.columns([1, 1])
        with col_img:
            if img_path.exists():
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.error(f"Image not found: {img_path}")

        with col_info:
            if has_gt and not pd.isna(row["FaceOcclusion"]):
                st.metric("GT Occlusion", f"{row['FaceOcclusion']:.4f}")
            st.metric("Predicted", f"{row['pred']:.4f}")
            if has_gt and not pd.isna(row["FaceOcclusion"]):
                st.metric("Delta (pred − GT)", f"{row['pred'] - row['FaceOcclusion']:+.4f}")
            if not pd.isna(row.get("gender", float("nan"))):
                st.write(f"**Gender:** {'Female' if row['gender'] == 0.0 else 'Male'}")

            st.divider()
            stars = load_stars()
            is_starred = any(s["split"] == split and s["filename"] == row["filename"] for s in stars)
            if is_starred:
                if st.button("★ Unstar"):
                    stars = [s for s in stars if not (s["split"] == split and s["filename"] == row["filename"])]
                    save_stars(stars)
                    st.rerun()
            else:
                if st.button("☆ Star this image"):
                    stars.append({"split": split, "filename": row["filename"]})
                    save_stars(stars)
                    st.rerun()


# ─── Stars ──────────────────────────────────────────────────────────────────────
with tab_stars:
    st.subheader("Starred Images")

    stars = load_stars()
    if not stars:
        st.info("No starred images yet — use the Picture tab to star images.")
    else:
        stars_df = pd.DataFrame(stars)
        split_stars = stars_df[stars_df["split"] == split] if "split" in stars_df.columns else stars_df

        if split_stars.empty:
            st.info(f"No starred images for the '{split}' split.")
        else:
            merged = split_stars.merge(df, on="filename", how="inner")
            merged = apply_filters(merged, gender_sel, occ_min, occ_max, has_gt)

            if merged.empty:
                st.info("No starred images match the current filters.")
            else:
                cols_per_row = 4
                for row_start in range(0, len(merged), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col_idx, (_, star_row) in enumerate(merged.iloc[row_start:row_start + cols_per_row].iterrows()):
                        with cols[col_idx]:
                            img_path = IMG_DIR / star_row["filename"]
                            if img_path.exists():
                                st.image(Image.open(img_path), use_container_width=True)
                            parts = []
                            if has_gt and not pd.isna(star_row.get("FaceOcclusion", float("nan"))):
                                parts.append(f"GT: {star_row['FaceOcclusion']:.3f}")
                            parts.append(f"Pred: {star_row['pred']:.3f}")
                            st.caption(" | ".join(parts))

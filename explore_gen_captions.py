r"""
Created on 22/6/2021 5:27 PM
@author: jiahuei

streamlit run explore_gen_captions.py
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from utils import METRICS, dict_filter, load_image_from_url


@st.cache(allow_output_mutation=True, max_entries=2)
def df_from_scores_detailed(scores_detailed) -> pd.DataFrame:
    scores_detailed = json.load(scores_detailed)
    scores = []
    for sc in scores_detailed:
        items = [sc["image_id"]]
        items += [sc[_] for _ in METRICS[:-1]]
        items += [sc["SPICE"]["All"]["f"]]
        scores.append(items)
    scores = pd.DataFrame(scores, columns=["image_id"] + METRICS)
    return scores


@st.cache(max_entries=2)
def df_from_captions(captions):
    captions = json.load(captions)
    captions = pd.DataFrame(captions)
    return captions


@st.cache(max_entries=2)
def merge_captions_scores(captions, scores_detailed):
    merged = captions.merge(scores_detailed, on="image_id", how="outer")
    assert len(captions) == len(scores_detailed)
    assert len(merged) == len(captions)
    assert len(merged) == len(scores_detailed)
    return merged


@st.cache(max_entries=2)
def load_coco_json(coco):
    coco = json.load(coco)
    images = [dict_filter(_, ["coco_url", "id"]) for _ in coco["images"]]
    captions = [dict_filter(_, ["caption", "image_id"]) for _ in coco["annotations"]]
    # Convert to DF
    images = pd.DataFrame(images)
    captions = pd.DataFrame(captions)
    # Group captions
    captions["caption"] = captions["caption"].map(lambda x: f"1. {x}")
    captions = captions.groupby("image_id").agg(lambda x: "\n\n".join(x)).reset_index()
    captions = captions.astype({"caption": "string"})
    # Merge and drop `id` column
    merged = images.merge(captions, how="outer", left_on="id", right_on="image_id")
    merged = merged[["coco_url", "image_id", "caption"]]
    return merged


def display_caption(df, key):
    st.header(key.capitalize())
    st.markdown(df[f"caption_{key}"])
    scores = [df[f"{m}_{key}"] for m in METRICS]
    scores = " | ".join(f"{s:.2f}" for s in scores)
    scores_md = f"| {key} | {scores} |\n"
    return scores_md


def main():
    st.sidebar.title("COCO Generated Caption Explorer")
    st.sidebar.markdown("---")

    # Top panel
    top1, top2, top3 = st.columns(3)
    with top1:
        seed = st.number_input(
            f"PRNG seed",
            min_value=0,
            max_value=None,
            value=0,
            step=1,
        )
    np.random.seed(seed)

    # COCO JSON for image URLs
    upload_help = "Provide MS-COCO validation JSON (captions_val2014.json)"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return
    else:
        coco_val = load_coco_json(uploaded_file)

    # Baseline
    upload_help = "Provide caption JSON (baseline)"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return
    else:
        baseline_captions = df_from_captions(uploaded_file)

    upload_help = "Provide detailed score JSON (baseline)"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return
    else:
        baseline_scores = df_from_scores_detailed(uploaded_file)
    baseline_scores["Random"] = np.random.random([len(baseline_scores)])

    baseline = merge_captions_scores(baseline_captions, baseline_scores)

    # Model
    upload_help = "Provide caption JSON (model)"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return
    else:
        model_captions = df_from_captions(uploaded_file)

    upload_help = "Provide detailed score JSON (model)"
    uploaded_file = st.sidebar.file_uploader(upload_help)
    if uploaded_file is None:
        st.info(f"{upload_help}, by uploading it in the sidebar")
        return
    else:
        model_scores = df_from_scores_detailed(uploaded_file)
    model_scores["Random"] = np.random.random([len(model_scores)])

    model = merge_captions_scores(model_captions, model_scores)

    # Merge DF
    merged = baseline.merge(
        model, on="image_id", how="outer", suffixes=["_baseline", "_model"]
    )
    merged = coco_val.merge(merged, on="image_id", how="outer").dropna(axis=0)
    merged = merged.rename(columns={"caption": "caption_coco"})
    assert len(baseline) == len(model)
    assert len(merged) == len(baseline)
    assert len(merged) == len(model)

    # Sort
    with top2:
        sort_by = baseline.columns.tolist()[2:]
        selected_sort = st.selectbox("Sort by", sort_by, sort_by.index("CIDEr"))
        relative_diff = st.checkbox("Relative difference")
    diff = merged[f"{selected_sort}_model"] - merged[f"{selected_sort}_baseline"]
    if relative_diff:
        diff /= merged[f"{selected_sort}_baseline"] + 1e-6
    sort_index = diff.sort_values(ascending=False).index
    sorted_df = merged.loc[sort_index]

    # Index selector
    with top3:
        selected_index = st.number_input(
            f"Jump to index: (0 - {len(sorted_df) - 1})",
            min_value=0,
            max_value=len(sorted_df) - 1,
            value=0,
            step=1,
        )
    sorted_df_selected = sorted_df.iloc[selected_index]

    col1, _, col2 = st.columns([4, 0.2, 6])
    # Display image
    with col1:
        st.header(f"Image ID: {sorted_df_selected['image_id']}")
        image = load_image_from_url(sorted_df_selected["coco_url"])
        st.image(image)
    with col2:
        score_md = "| Approach | " + " | ".join(METRICS) + " | "
        score_md += """
        | --- | --- |
        """
        score_md += display_caption(sorted_df_selected, "baseline")
        score_md += display_caption(sorted_df_selected, "model")
        st.header("Scores")
        st.markdown(score_md)

    st.header("Ground Truth")
    st.markdown(sorted_df_selected["caption_coco"])

    st.markdown("---")
    with st.beta_expander("Debugging info"):
        st.subheader("Merged Dataframe")
        st.dataframe(merged.iloc[:100])
        st.subheader("Baseline Dataframe")
        st.dataframe(baseline.iloc[:100])
        st.subheader("Model Dataframe")
        st.dataframe(model.iloc[:100])


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()

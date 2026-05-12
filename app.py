"""
app.py
======
Streamlit front-end for the pneumonia detection + RAG clinical report pipeline.

Run:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
)


# ── Load model once per session ───────────────────────────────────────────────
# @st.cache_resource ensures the model is only loaded from disk once,
# even if Streamlit re-runs the script on every user interaction.

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    from pipeline import load_model, MODEL_PATH
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 About this tool")
    st.markdown(
        """
        **Model** Fine-tuned ResNet18  
        **Dataset** Chest X-Ray Images (Pneumonia)  
        &nbsp;&nbsp;Guangzhou Women & Children's Medical Center  
        &nbsp;&nbsp;via [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

        | Metric | Score |
        |---|---|
        | Test accuracy | 90.7 % |
        | Normal recall | 80.8 % |
        | Pneumonia recall | 96.7 % |

        
        **Author** Aye Nyein Kyaw
        """
    )
    st.divider()
    st.warning(
        "⚠️ **Educational use only.**  \n"
        "This tool does not constitute medical advice and must not "
        "replace qualified clinical or radiological judgment."
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        "1. ResNet18 inference  \n"
        "2. Grad-CAM attention map  \n"
        "3. ChromaDB guideline retrieval  \n"
        "4. Claude API report synthesis"
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection — RAG Clinical Report")
st.caption("Upload a chest X-ray to get a model prediction, attention heatmap, and AI-generated clinical summary.")

# ── Model check ───────────────────────────────────────────────────────────────
model = get_model()

if model is None:
    st.error(
        "**Model file not found.**  \n"
        "Expected path: `models/finetuned_resnet.pth`  \n"
        "Move your ResNet18 checkpoint there and restart."
    )
    st.stop()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a chest X-ray (JPEG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload an X-ray image above to begin.")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running inference, Grad-CAM, and report generation…"):
    from pipeline import run_pipeline, CONFIDENCE_THRESHOLD
    result = run_pipeline(image, model)

pred             = result["prediction"]
confidence       = result["confidence"]
probs            = result["probs"]
overlay          = result["overlay"]
gradcam_desc     = result["gradcam_desc"]
inconclusive     = result["inconclusive"]
report           = result["report"]
guideline_chunks = result["guideline_chunks"]


# ── Image columns ─────────────────────────────────────────────────────────────
st.divider()
col_img, col_cam = st.columns(2)

with col_img:
    st.subheader("Original X-ray")
    st.image(image, use_container_width=True)

with col_cam:
    st.subheader("Grad-CAM attention map")
    st.image(overlay, use_container_width=True)
    st.caption(f"Model attention: *{gradcam_desc}*")


# ── Prediction ────────────────────────────────────────────────────────────────
st.divider()

if inconclusive:
    st.warning(
        f"**Inconclusive** — model confidence {confidence:.1%} is below the "
        f"{CONFIDENCE_THRESHOLD:.0%} threshold required to generate a report.  \n"
        "Please review the image with a qualified clinician."
    )
else:
    # Colour-coded prediction badge
    badge_colour = "#c62828" if pred == "PNEUMONIA" else "#2e7d32"
    st.markdown(
        f'<div style="display:inline-block;background:{badge_colour};color:#fff;'
        f'padding:10px 22px;border-radius:8px;font-size:20px;font-weight:600">'
        f'{pred} &nbsp;·&nbsp; {confidence:.1%}</div>',
        unsafe_allow_html=True,
    )

# Class probability bars
st.markdown("#### Class probabilities")
pb_col1, pb_col2 = st.columns(2)
with pb_col1:
    st.metric("NORMAL",    f"{probs[0]:.1%}")
    st.progress(float(probs[0]))
with pb_col2:
    st.metric("PNEUMONIA", f"{probs[1]:.1%}")
    st.progress(float(probs[1]))


# ── Clinical report ───────────────────────────────────────────────────────────
if report and not inconclusive:
    st.divider()
    st.subheader("📋 AI Clinical Report")

    if "error" in report:
        st.error(f"Report generation failed: {report['error']}")

    else:
        # Urgency badge
        urgency       = report.get("urgency_level", "ROUTINE")
        urgency_icons = {"ROUTINE": "🟢", "ELEVATED": "🟡", "URGENT": "🔴"}
        urgency_icon  = urgency_icons.get(urgency, "⚪")
        st.markdown(f"**Urgency level:** {urgency_icon} {urgency}")

        # Assessment
        st.markdown(f"**Assessment:** {report.get('assessment', '—')}")

        # Observations
        with st.expander("Key observations", expanded=True):
            for obs in report.get("key_observations", []):
                st.markdown(f"- {obs}")

        # Recommended actions
        with st.expander("Recommended next steps"):
            for step in report.get("recommended_next_steps", []):
                st.markdown(f"- {step}")

        # Guideline points that were used
        with st.expander("Relevant guideline points"):
            for point in report.get("relevant_guideline_points", []):
                st.markdown(f"- {point}")

        # Sources
        if guideline_chunks:
            with st.expander("Retrieved sources"):
                for chunk in guideline_chunks:
                    st.markdown(
                        f"**{chunk['source']}** — *{chunk['section']}*  \n"
                        f"{chunk['text'][:280]}…"
                    )

        # Disclaimer
        st.info(report.get("disclaimer", "For educational use only."))

        # Raw JSON for debugging
        with st.expander("Raw JSON response"):
            st.json(report)

elif not inconclusive and report is None:
    st.info(
        "Report skipped — ChromaDB knowledge base is empty.  \n"
        "Run `python setup_knowledge_base.py` once to index the guidelines."
    )

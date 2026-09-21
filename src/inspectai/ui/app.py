from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from inspectai.inference import Predictor
from inspectai.model_registry import available_models
from inspectai.monitoring import monitoring_summary
from inspectai.storage import PredictionStore

st.set_page_config(page_title="InspectAI", page_icon="🔎", layout="wide")
st.title("InspectAI — Visual Defect Detection")
st.caption("Binary visual defect classification with confidence and Grad-CAM explanations")


@st.cache_resource
def load_predictor(checkpoint: str) -> Predictor:
    return Predictor(Path(checkpoint))


@st.cache_resource
def load_store() -> PredictionStore:
    return PredictionStore(os.getenv("INSPECTAI_DB_PATH", "logs/inspectai.db"))


models = available_models()
if not models:
    st.error("No trained model is available. Run the training command first.")
    st.stop()
model_names = list(models)
selected_model = st.selectbox(
    "Prediction model",
    model_names,
    format_func=lambda name: models[name]["display_name"],
    help="Only models with an existing trained checkpoint are shown.",
)
uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    try:
        predictor = load_predictor(models[selected_model]["checkpoint"])
        with st.spinner("Inspecting image…"):
            result = predictor.predict(image, explain=True)
        prediction_id = load_store().add(uploaded.name, result)
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            st.image(result.heatmap, caption="Grad-CAM influence map", use_container_width=True)
        status = "✅ NORMAL" if result.label == "normal" else "⚠️ DEFECTIVE"
        st.subheader(status)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", f"{result.confidence:.1%}")
        c2.metric("Latency", f"{result.latency_ms:.1f} ms")
        c3.metric("Model", models[selected_model]["display_name"])
        c4.metric("Model version", result.model_version)
        st.bar_chart(result.probabilities)
        feedback = st.radio("Was this prediction useful?", ["correct", "incorrect", "unsure"], horizontal=True)
        if st.button("Save feedback"):
            load_store().feedback(prediction_id, feedback)
            st.success("Feedback saved locally.")
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("make quick", language="bash")
else:
    st.info("Upload an image to start. The model is a demonstration aid, not a production quality gate.")

with st.sidebar:
    st.header("Recent monitoring")
    summary = monitoring_summary(load_store())
    st.metric("Predictions", summary["prediction_count"])
    if summary.get("mean_latency_ms") is not None:
        st.metric("Mean latency", f"{summary['mean_latency_ms']:.1f} ms")
        st.metric("Defective rate", f"{summary['defective_prediction_rate']:.1%}")
    if summary.get("feedback_accuracy") is not None:
        st.metric("Feedback accuracy", f"{summary['feedback_accuracy']:.1%}")
    st.caption("Local rolling metrics; feedback can be biased and is not a replacement for labeled evaluation.")

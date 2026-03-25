import streamlit as st
import requests
import pandas as pd
from PIL import Image

API_DEFAULT = "http://localhost:8000"

st.set_page_config(
    page_title="Indo Fashion Classifier",
    page_icon="👗",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
api_url = st.sidebar.text_input("API URL", value=API_DEFAULT)

try:
    health = requests.get(f"{api_url}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.sidebar.success("API is running and model is loaded.")
    else:
        st.sidebar.warning("API is running but model is NOT loaded.")
except requests.exceptions.ConnectionError:
    st.sidebar.error(
        "Cannot reach the API. Start it with:\n\n"
        "`uvicorn src.api:app --port 8000` (from project root)\n\n"
        "or\n\n"
        "`uvicorn api:app --port 8000` (from src folder)"
    )
except Exception as e:
    st.sidebar.error(f"API error: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Indo Fashion Classifier**\n\n"
    "Upload an image of Indian clothing and the model "
    "will predict which of the 15 categories it belongs to."
)

# ── Main area ────────────────────────────────────────────────────────────────
st.title("Indo Fashion Classifier")
st.markdown(
    "Upload an image of Indian fashion (saree, kurta, lehenga, etc.) "
    "to get an instant classification with confidence scores."
)

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        with st.spinner("Classifying..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{api_url}/predict", files=files, timeout=30)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    "Make sure it is running with "
                    "`uvicorn src.api:app --port 8000` (from project root) "
                    "or `uvicorn api:app --port 8000` (from src folder)."
                )
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        st.metric(
            label="Predicted Class",
            value=predicted_class.replace("_", " ").title(),
        )
        st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

        st.markdown("---")

        # Top-5 bar chart
        st.subheader("Top 5 Predictions")
        probs = result["all_probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_probs[:5]
        top5_df = pd.DataFrame(top5, columns=["Class", "Probability"])
        top5_df["Class"] = top5_df["Class"].str.replace("_", " ").str.title()
        top5_df = top5_df.set_index("Class")
        st.bar_chart(top5_df)

        # Full table (expandable)
        with st.expander("All 15 class probabilities"):
            full_df = pd.DataFrame(sorted_probs, columns=["Class", "Probability"])
            full_df["Class"] = full_df["Class"].str.replace("_", " ").str.title()
            full_df["Probability"] = full_df["Probability"].apply(lambda x: f"{x * 100:.4f}%")
            full_df.index = range(1, len(full_df) + 1)
            st.table(full_df)

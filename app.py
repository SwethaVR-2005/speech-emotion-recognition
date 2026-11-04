
import streamlit as st
import numpy as np
import librosa
import pickle
import io
import soundfile as sf

from pathlib import Path


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model.pkl"

SAMPLE_RATE = 22050
N_MFCC = 40

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "emotion_model.pkl not found. Train your model first."
        )

    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    # Supports the dictionary saved by the new train_model.py
    if isinstance(model_data, dict):
        model = model_data["model"]
        expected_features = model_data.get("feature_count", 80)
        sample_rate = model_data.get("sample_rate", SAMPLE_RATE)
        n_mfcc = model_data.get("n_mfcc", N_MFCC)
    else:
        # Compatibility with an older model file
        model = model_data
        expected_features = 80
        sample_rate = SAMPLE_RATE
        n_mfcc = N_MFCC

    return model, expected_features, sample_rate, n_mfcc


# --------------------------------------------------
# 3. EXTRACT FEATURES
# --------------------------------------------------

def extract_features(audio, sr):

    # Use a separate analysis copy.
    # This does not modify the original recording.
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError("The audio is empty.")

    # Resample only for model analysis.
    # No pitch shifting or time stretching.
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features.reshape(1, -1)


# --------------------------------------------------
# 4. READ AUDIO FOR ANALYSIS
# --------------------------------------------------

def read_audio_for_analysis(audio_bytes):

    audio, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=False
    )

    # Convert stereo to mono for prediction only.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return audio, sr


# --------------------------------------------------
# 5. PREDICT EMOTION
# --------------------------------------------------

def predict_emotion(audio_bytes, model, expected_features):

    audio, sr = read_audio_for_analysis(audio_bytes)

    features = extract_features(audio, sr)

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: expected {expected_features}, "
            f"received {features.shape[1]}."
        )

    prediction = model.predict(features)[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    return prediction, probabilities


# --------------------------------------------------
# 6. EMOTION DISPLAY
# --------------------------------------------------

EMOTION_EMOJIS = {
    "NEUTRAL": "😐",
    "CALM": "😌",
    "HAPPY": "😊",
    "SAD": "😢",
    "ANGRY": "😠",
    "FEARFUL": "😨",
    "DISGUST": "🤢",
    "SURPRISED": "😲"
}


# --------------------------------------------------
# 7. MAIN APPLICATION
# --------------------------------------------------

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Record your voice or upload an audio file "
    "to analyze the vocal emotion."
)

st.caption(
    "Your original audio is preserved for playback. "
    "Audio analysis does not apply pitch or speed changes."
)


# Load model
try:
    model, expected_features, model_sr, model_mfcc = load_model()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


if model_sr != SAMPLE_RATE or model_mfcc != N_MFCC:
    st.error(
        "The model settings do not match this app. "
        "Please check your training and app settings."
    )
    st.stop()


# --------------------------------------------------
# 8. LIVE RECORDING
# --------------------------------------------------

st.subheader("🎤 Record your voice")

recorded_audio = st.audio_input(
    "Click to start recording"
)

if recorded_audio is not None:

    audio_bytes = recorded_audio.getvalue()

    st.success("Recording received!")

    # Play the original recording without processing.
    st.subheader("Original recording")
    st.audio(audio_bytes)

    if st.button("Analyze recorded voice", key="analyze_live"):

        try:
            with st.spinner("Analyzing your voice..."):

                emotion, probabilities = predict_emotion(
                    audio_bytes,
                    model,
                    expected_features
                )

            emoji = EMOTION_EMOJIS.get(emotion, "🎭")

            st.subheader("Prediction")
            st.markdown(f"## {emoji} {emotion}")

            if probabilities is not None:

                st.subheader("Prediction confidence")

                classes = model.classes_

                for label, probability in sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):

                    st.write(
                        f"{label}: {probability * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(probability, 0, 1))
                    )

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# --------------------------------------------------
# 9. UPLOAD AUDIO
# --------------------------------------------------

st.divider()

st.subheader("📁 Upload an audio file")

uploaded_audio = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)

if uploaded_audio is not None:

    audio_bytes = uploaded_audio.getvalue()

    st.subheader("Original uploaded audio")
    st.audio(audio_bytes)

    if st.button("Analyze uploaded audio", key="analyze_upload"):

        try:
            with st.spinner("Analyzing uploaded audio..."):

                emotion, probabilities = predict_emotion(
                    audio_bytes,
                    model,
                    expected_features
                )

            emoji = EMOTION_EMOJIS.get(emotion, "🎭")

            st.subheader("Prediction")
            st.markdown(f"## {emoji} {emotion}")

            if probabilities is not None:

                st.subheader("Prediction confidence")

                for label, probability in sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):

                    st.write(
                        f"{label}: {probability * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(probability, 0, 1))
                    )

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Note: This system estimates emotion from vocal characteristics. "
    "It cannot determine a person's actual feelings with certainty."
)

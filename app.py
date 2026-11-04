
import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import librosa
import joblib
import queue
import threading

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh


# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_PATH = "emotion_model.pkl"

TARGET_SR = 22050
MAX_DURATION = 10

# Must match your training feature extraction
N_MFCC = 40
FEATURE_DURATION = 3


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)


# =====================================================
# 3. AUDIO STORAGE
# =====================================================

class AudioStorage:

    def __init__(self):

        self.audio_queue = queue.Queue()

        self.lock = threading.Lock()

        self.active = False

        self.frame_count = 0

        self.latest_rms = 0.0

    def start(self):

        with self.lock:

            self.active = True

            self.frame_count = 0

            self.latest_rms = 0.0

        while not self.audio_queue.empty():

            try:
                self.audio_queue.get_nowait()

            except queue.Empty:
                break

    def stop(self):

        with self.lock:
            self.active = False

    def is_active(self):

        with self.lock:
            return self.active

    def update_status(self, rms):

        with self.lock:

            self.frame_count += 1

            self.latest_rms = rms

    def get_status(self):

        with self.lock:

            return self.frame_count, self.latest_rms


@st.cache_resource
def get_audio_storage():

    return AudioStorage()


audio_storage = get_audio_storage()


# =====================================================
# 4. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

    MODEL_LOADED = True

except Exception as e:

    MODEL_LOADED = False

    model = None

    st.error(f"Model loading failed: {e}")


# =====================================================
# 5. FEATURE EXTRACTION
# =====================================================

def extract_features(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(audio)

    # Training used 3 seconds starting at 0.5 sec
    start = int(0.5 * TARGET_SR)

    end = int(3.5 * TARGET_SR)

    audio = audio[start:end]

    required_samples = (
        FEATURE_DURATION * TARGET_SR
    )

    if len(audio) < required_samples:

        audio = np.pad(
            audio,
            (0, required_samples - len(audio))
        )

    elif len(audio) > required_samples:

        audio = audio[:required_samples]

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC
    )

    # 40 MFCC means + 40 standard deviations
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features.reshape(1, -1)


# =====================================================
# 6. PREDICTION FUNCTION
# =====================================================

def predict_emotion(audio):

    if not MODEL_LOADED:

        raise RuntimeError(
            "emotion_model.pkl could not be loaded."
        )

    features = extract_features(audio)

    if features.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Feature mismatch! Model expects "
            f"{model.n_features_in_}, "
            f"but received {features.shape[1]}."
        )

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    return prediction, probabilities, confidence


# =====================================================
# 7. SESSION STATE
# =====================================================

defaults = {

    "recording": False,

    "recording_finished": False,

    "recorded_audio": [],

    "sample_count": 0,

    "original_sample_rate": None,

    "last_recording": None,

    "prediction": None,

    "probabilities": None,

    "confidence": 0.0,

    "speech_id": 0,

    "audio_source": None,

    "error_message": None
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =====================================================
# 8. RESET RESULTS
# =====================================================

def reset_results():

    st.session_state.recording = False

    st.session_state.recording_finished = False

    st.session_state.recorded_audio = []

    st.session_state.sample_count = 0

    st.session_state.original_sample_rate = None

    st.session_state.last_recording = None

    st.session_state.prediction = None

    st.session_state.probabilities = None

    st.session_state.confidence = 0.0

    st.session_state.error_message = None


# =====================================================
# 9. MICROPHONE CALLBACK
# =====================================================

def create_audio_callback(storage):

    def audio_callback(frame):

        audio = frame.to_ndarray()

        # Convert stereo to mono
        if audio.ndim == 2:

            audio = np.mean(
                audio,
                axis=0
            )

        # Convert integer audio correctly
        if np.issubdtype(
            audio.dtype,
            np.integer
        ):

            max_value = np.iinfo(
                audio.dtype
            ).max

            audio = (
                audio.astype(np.float32)
                / max_value
            )

        else:

            audio = audio.astype(np.float32)

        audio = np.nan_to_num(audio)

        # Calculate microphone signal level
        rms = float(
            np.sqrt(np.mean(audio ** 2))
        )

        storage.update_status(rms)

        # Store audio at ORIGINAL sample rate
        if storage.is_active():

            storage.audio_queue.put(
                (
                    audio.copy(),
                    frame.sample_rate
                )
            )

        return frame

    return audio_callback


# =====================================================
# 10. HEADER
# =====================================================

st.title(
    "🎙️ Speech Emotion Recognition"
)

st.write(
    "AI-powered emotion detection using voice."
)

st.info(
    "Available emotions depend on the classes "
    "used while training your model."
)


# =====================================================
# 11. SELECT INPUT METHOD
# =====================================================

input_method = st.radio(
    "Choose Audio Input",
    [
        "Live Microphone",
        "Upload Original WAV File"
    ],
    horizontal=True
)


# =====================================================
# 12. ORIGINAL AUDIO UPLOAD
# =====================================================

if input_method == "Upload Original WAV File":

    st.subheader("Upload Original Audio")

    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        try:

            audio, original_sr = librosa.load(
                uploaded_file,
                sr=None,
                mono=True
            )

            st.success(
                "Original audio loaded successfully!"
            )

            st.write(
                f"Original sample rate: {original_sr} Hz"
            )

            st.write(
                f"Audio duration: {len(audio) / original_sr:.2f} seconds"
            )

            st.write(
                f"Audio samples: {len(audio)}"
            )

            st.subheader("Original Audio Playback")

            st.audio(
                uploaded_file.getvalue(),
                format="audio/wav"
            )

            if st.button(
                "Analyze Original Audio",
                type="primary"
            ):

                reset_results()

                # Resample complete file only once
                if original_sr != TARGET_SR:

                    audio = librosa.resample(
                        audio,
                        orig_sr=original_sr,
                        target_sr=TARGET_SR
                    )

                st.session_state.last_recording = audio

                st.session_state.audio_source = "upload"

                prediction, probabilities, confidence = (
                    predict_emotion(audio)
                )

                st.session_state.prediction = prediction

                st.session_state.probabilities = probabilities

                st.session_state.confidence = confidence

                st.session_state.speech_id += 1

        except Exception as e:

            st.error(
                f"Audio processing error: {e}"
            )


# =====================================================
# 13. LIVE MICROPHONE
# =====================================================

elif input_method == "Live Microphone":

    st.subheader("Live Microphone Input")

    ctx = webrtc_streamer(

        key="speech-emotion-microphone",

        mode=WebRtcMode.SENDONLY,

        audio_frame_callback=create_audio_callback(
            audio_storage
        ),

        media_stream_constraints={
            "audio": True,
            "video": False
        },

        async_processing=True
    )

    # Refresh microphone status while connected
    if ctx.state.playing and not st.session_state.recording:

        st_autorefresh(
            interval=500,
            key="microphone_status_refresh"
        )

    # Microphone diagnostics
    frame_count, latest_rms = (
        audio_storage.get_status()
    )

    st.subheader("Microphone Diagnostics")

    if ctx.state.playing:

        st.success("Microphone connected")

        st.metric(
            "Audio Frames Received",
            frame_count
        )

        st.progress(
            min(latest_rms * 10, 1.0),
            text=f"Audio signal: {latest_rms:.5f}"
        )

        if frame_count == 0:

            st.warning(
                "Waiting for microphone audio..."
            )

        elif latest_rms < 0.001:

            st.warning(
                "Audio signal is very low. "
                "Check your microphone."
            )

        else:

            st.success(
                "Microphone audio is reaching Python!"
            )

    else:

        st.warning(
            "Click START above and allow microphone access."
        )

    # Start recording
    if (
        ctx.state.playing
        and not st.session_state.recording
    ):

        if st.button(
            "Start 10-Second Recording",
            type="primary"
        ):

            reset_results()

            audio_storage.start()

            st.session_state.recording = True

            st.session_state.audio_source = "microphone"

            st.rerun()

    # Collect audio
    if st.session_state.recording:

        st_autorefresh(
            interval=250,
            key="recording_refresh"
        )

        while not audio_storage.audio_queue.empty():

            try:

                audio, original_sr = (
                    audio_storage.audio_queue.get_nowait()
                )

            except queue.Empty:

                break

            # Save the first sample rate
            if st.session_state.original_sample_rate is None:

                st.session_state.original_sample_rate = (
                    original_sr
                )

            # Ensure all frames have same sample rate
            if original_sr != st.session_state.original_sample_rate:

                st.session_state.error_message = (
                    "Microphone sample rate changed during recording."
                )

                audio_storage.stop()

                st.session_state.recording = False

                break

            sr = st.session_state.original_sample_rate

            remaining_samples = (
                MAX_DURATION * sr
                - st.session_state.sample_count
            )

            if remaining_samples <= 0:

                break

            audio = audio[:remaining_samples]

            st.session_state.recorded_audio.append(
                audio
            )

            st.session_state.sample_count += len(audio)

        if st.session_state.original_sample_rate:

            sr = st.session_state.original_sample_rate

            duration = (
                st.session_state.sample_count / sr
            )

        else:

            duration = 0

        duration = min(
            duration,
            MAX_DURATION
        )

        remaining = MAX_DURATION - duration

        st.success(
            "🔴 Recording in progress..."
        )

        st.progress(
            duration / MAX_DURATION,
            text=f"Recording: {duration:.1f} / 10 seconds"
        )

        st.write(
            f"Time remaining: {remaining:.1f} seconds"
        )

        # Stop after 10 seconds
        if duration >= MAX_DURATION:

            audio_storage.stop()

            st.session_state.recording = False

            st.session_state.recording_finished = True

            st.rerun()

    # Analyze recorded audio
    if st.session_state.recording_finished:

        if st.session_state.recorded_audio:

            if st.session_state.prediction is None:

                try:

                    with st.spinner(
                        "Analyzing your voice..."
                    ):

                        audio = np.concatenate(
                            st.session_state.recorded_audio
                        )

                        original_sr = (
                            st.session_state.original_sample_rate
                        )

                        # Resample the COMPLETE recording once
                        if original_sr != TARGET_SR:

                            audio = librosa.resample(
                                audio,
                                orig_sr=original_sr,
                                target_sr=TARGET_SR
                            )

                        st.session_state.last_recording = audio

                        prediction, probabilities, confidence = (
                            predict_emotion(audio)
                        )

                        st.session_state.prediction = prediction

                        st.session_state.probabilities = probabilities

                        st.session_state.confidence = confidence

                        st.session_state.speech_id += 1

                except Exception as e:

                    st.session_state.error_message = str(e)

        else:

            st.session_state.error_message = (
                "No audio was recorded. Check your microphone."
            )


# =====================================================
# 14. DISPLAY AUDIO
# =====================================================

if st.session_state.last_recording is not None:

    st.divider()

    st.subheader("Recorded / Processed Audio")

    st.audio(
        st.session_state.last_recording,
        sample_rate=TARGET_SR
    )

    st.write(
        f"Processed sample rate: {TARGET_SR} Hz"
    )

    st.write(
        f"Processed duration: "
        f"{len(st.session_state.last_recording) / TARGET_SR:.2f} seconds"
    )


# =====================================================
# 15. DISPLAY PREDICTION
# =====================================================

if st.session_state.prediction is not None:

    emotion = str(
        st.session_state.prediction
    ).upper()

    st.divider()

    st.subheader("AI Emotion Detection Result")

    st.success(
        f"YOU ARE {emotion}"
    )

    st.metric(
        "Prediction Confidence",
        f"{st.session_state.confidence:.2f}%"
    )

    st.subheader("Emotion Probabilities")

    for emotion_name, probability in zip(
        model.classes_,
        st.session_state.probabilities
    ):

        st.write(
            f"{emotion_name}: {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )

    # Voice output
    components.html(

        f"""
        <script>

        const speechId = "{st.session_state.speech_id}";

        const emotion = "{emotion}";

        if (window.parent.lastSpeechId !== speechId) {{

            window.parent.lastSpeechId = speechId;

            const message = new SpeechSynthesisUtterance(
                "YOU ARE " + emotion
            );

            message.lang = "en-US";

            message.rate = 0.9;

            window.parent.speechSynthesis.cancel();

            window.parent.speechSynthesis.speak(message);
        }}

        </script>
        """,

        height=0
    )


# =====================================================
# 16. ERROR DISPLAY
# =====================================================

if st.session_state.error_message:

    st.error(
        st.session_state.error_message
    )


# =====================================================
# 17. RECORD AGAIN
# =====================================================

if (
    st.session_state.recording_finished
    or st.session_state.prediction is not None
):

    if st.button("Start New Analysis"):

        reset_results()

        audio_storage.stop()

        st.rerun()

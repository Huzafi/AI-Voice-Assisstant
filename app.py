import os
import io
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# -------------------------
# API Key setup
# -------------------------
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]


if not GEMINI_KEY:
    st.error("Google API key not found. Please set it in .env or Streamlit Secrets.")
else:
    genai.configure(api_key=GEMINI_KEY)

try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    # fallback model
    model = genai.GenerativeModel("gemini-1.5-pro")

# -------------------------
# Voice input processor
# -------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text = ""

    def recv_audio(self, frame):
        # Convert WebRTC audio frame to sr.AudioData
        audio_data = frame.to_ndarray().flatten().astype("int16")
        audio = sr.AudioData(audio_data.tobytes(), frame.sample_rate, 2)
        try:
            self.text = self.recognizer.recognize_google(audio)
        except Exception:
            pass
        return frame


def voice_input():
    ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    if ctx.audio_processor:
        return ctx.audio_processor.text
    return ""


# -------------------------
# LLM logic
# -------------------------
def llm_model_object(user_text: str, max_tokens: int = 1024) -> str:
    if not user_text or not user_text.strip():
        return "Please provide a question or text to explain."

    lower_text = user_text.lower()

    # Default = brief response
    prompt = (
        f"QUESTION OR TOPIC: {user_text}\n\n"
        "Answer briefly in 2-3 sentences."
    )

    if "long" in lower_text or "detailed" in lower_text:
        prompt = (
            "You are a helpful, thorough assistant. "
            "Provide a detailed, structured explanation with sections, examples, "
            "and short code snippets where relevant.\n\n"
            f"QUESTION OR TOPIC: {user_text}\n\n"
            "Answer in detail:"
        )
    elif "briefly" in lower_text:
        prompt = (
            f"QUESTION OR TOPIC: {user_text}\n\n"
            "Answer very briefly in 1-2 sentences."
        )

    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )

    result = ""
    if hasattr(response, "text") and response.text:
        result = response.text
    else:
        cand = getattr(response, "candidates", None)
        if cand and len(cand) > 0:
            first = cand[0]
            result = (
                getattr(first, "content", None)
                or getattr(first, "text", None)
                or getattr(first, "output", None)
                or str(first)
            )
        else:
            result = str(response)

    return result.strip()


# -------------------------
# TTS helper
# -------------------------
def text_to_speech_bytes(text: str) -> io.BytesIO:
    if not text or not text.strip():
        text = "Sorry, I couldn't generate a response."

    mp3_fp = io.BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp


# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="AS AI Voice Assistant", layout="centered")
    st.title("AI Voice Assistant 🤖")

    mode = st.radio("Select input method:", ["Text", "Voice"])

    user_text = ""
    if mode == "Text":
        user_text = st.text_input("Type your question:")
    else:
        st.info("🎤 Click Start below and speak...")
        user_text = voice_input()
        if user_text:
            st.write("🗣 You said:", user_text)

    if user_text:
        with st.spinner("Generating response..."):
            response = llm_model_object(user_text, max_tokens=1024)
            st.text_area("Response:", value=response, height=350)

            mp3_fp = text_to_speech_bytes(response)
            audio_bytes = mp3_fp.getvalue()

            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="Download Speech",
                data=audio_bytes,
                file_name="response.mp3",
                mime="audio/mp3"
            )


if __name__ == "__main__":
    main()

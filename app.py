import os
import io
import tempfile
import speech_recognition as sr
import streamlit as st
import google.generativeai as genai
from elevenlabs import ElevenLabs

# -------------------------
# Configure API keys
# -------------------------
GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
ELEVENLABS_KEY = st.secrets["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE = st.secrets["ELEVENLABS_VOICE_ID"]  # e.g., "21m00Tcm4TlvDq8ikWAM"

if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env file")
if not ELEVENLABS_KEY or not ELEVENLABS_VOICE:
    raise RuntimeError("Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in your .env file")

genai.configure(api_key=GEMINI_KEY)
eleven = ElevenLabs(api_key=ELEVENLABS_KEY)

# -------------------------
# Helper functions
# -------------------------
def voice_input_streamlit():
    r = sr.Recognizer()
    audio_file = st.file_uploader("Upload your voice (wav/mp3):", type=["wav", "mp3"])

    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        with sr.AudioFile(temp_audio_path) as source:
            audio = r.record(source)
            try:
                text = r.recognize_google(audio)
                st.write("🗣 You said:", text)
                return text
            except sr.UnknownValueError:
                st.error("❌ Could not understand audio")
                return ""
            except sr.RequestError as e:
                st.error(f"⚠️ STT request error: {e}")
                return ""
    return ""

def llm_model_object(user_text: str, max_tokens: int = 1024) -> str:
    if not user_text or not user_text.strip():
        return "Please provide a question or text to explain."

    lower_text = user_text.lower()
    prompt = f"QUESTION OR TOPIC: {user_text}\n\nAnswer briefly in 2-3 sentences."

    if "long" in lower_text or "detailed" in lower_text:
        prompt = (
            "You are a helpful, thorough assistant. Provide a detailed, structured explanation with sections, examples, "
            "and short code snippets where relevant.\n\n"
            f"QUESTION OR TOPIC: {user_text}\n\nAnswer in detail:"
        )
    elif "briefly" in lower_text:
        prompt = f"QUESTION OR TOPIC: {user_text}\n\nAnswer very briefly in 1-2 sentences."

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=0.3,
        ),
    )

    if hasattr(response, "text") and response.text:
        return response.text.strip()
    elif getattr(response, "candidates", None):
        first = response.candidates[0]
        return getattr(first, "content", getattr(first, "text", str(first))).strip()
    else:
        return str(response).strip()

def text_to_speech_bytes_elevenlabs(text: str) -> io.BytesIO:
    if not text or not text.strip():
        text = "Sorry, I couldn't generate a response."

    # use the client instance
    audio_data = eleven.generate(
        text=text,
        voice=ELEVENLABS_VOICE,
        model="eleven_multilingual_v1"
    )

    mp3_fp = io.BytesIO(audio_data)
    mp3_fp.seek(0)
    return mp3_fp


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="AS AI Voice Assistant", layout="centered")
    st.title("AS AI Voice Assistant 🤖")

    mode = st.radio("Select input method:", ["Text", "Voice"])
    user_text = st.text_input("Type your question:") if mode == "Text" else voice_input_streamlit()

    if user_text:
        with st.spinner("Generating response..."):
            response = llm_model_object(user_text, max_tokens=1024)
            st.text_area("Response:", value=response, height=350)

            # ElevenLabs TTS
            mp3_fp = text_to_speech_bytes_elevenlabs(response)
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

import os
import io
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import streamlit as st


GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env file")
genai.configure(api_key=GEMINI_KEY)

# -------------------------
# Helper functions
# -------------------------
def voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            r.adjust_for_ambient_noise(source, duration=0.4)
            audio = r.listen(source, phrase_time_limit=12)
        text = r.recognize_google(audio)
        print("🗣 You said:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError as e:
        print("⚠️ STT request error:", e)
        return ""


def llm_model_object(user_text: str, max_tokens: int = 1024) -> str:
    """
    Smart prompt for Gemini: decides whether to give short or detailed answers
    based on user keywords like 'long', 'detailed', or 'briefly'.
    """

    if not user_text or not user_text.strip():
        return "Please provide a question or text to explain."

    lower_text = user_text.lower()

    # Default = brief response
    prompt = (
        f"QUESTION OR TOPIC: {user_text}\n\n"
        "Answer briefly in 2-3 sentences."
    )

    # If user asks for long/detailed
    if "long" in lower_text or "detailed" in lower_text:
        prompt = (
            "You are a helpful, thorough assistant. "
            "Provide a detailed, structured explanation with sections, examples, "
            "and short code snippets where relevant.\n\n"
            f"QUESTION OR TOPIC: {user_text}\n\n"
            "Answer in detail:"
        )

    # If user asks for briefly
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

    # Robust extraction
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



def text_to_speech_bytes(text: str) -> io.BytesIO:
    """
    Convert text to mp3 bytes in memory and return a BytesIO object.
    """
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
    st.title("AS AI Voice Assistant 🤖")

    mode = st.radio("Select input method:", ["Text", "Voice"])

    user_text = ""
    if mode == "Text":
        user_text = st.text_input("Type your question:")
    else:
        if st.button("🎤 Speak"):
            with st.spinner("Listening..."):
                user_text = voice_input()

    if user_text:
        with st.spinner("Generating response..."):
            response = llm_model_object(user_text, max_tokens=1024)
            st.text_area("Response:", value=response, height=350)

            # Convert to mp3 in-memory and play / download
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

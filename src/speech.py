"""
SmartFuel AI — Speech Recognition Module (Optional)
=====================================================
Provides optional speech-to-text input for the AI Manager Assistant.

This module uses the SpeechRecognition library with the Google Web Speech API
for recognition.  The feature is OPTIONAL — if the library is unavailable or
the microphone is not accessible, the application continues to work normally
with text input only.

Dependency
----------
    pip install SpeechRecognition pyaudio

pyaudio is a C-extension and may require:
    Windows : pip install pipwin && pipwin install pyaudio
    macOS   : brew install portaudio && pip install pyaudio
    Linux   : sudo apt-get install python3-pyaudio

If pyaudio cannot be installed, set SPEECH_ENABLED = False.  The Streamlit
app checks SPEECH_ENABLED before rendering the microphone button.
"""

import io

# ── Availability check ────────────────────────────────────────────────────
SPEECH_ENABLED = False
_IMPORT_ERROR   = ""

try:
    import speech_recognition as sr
    SPEECH_ENABLED = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def get_status() -> dict:
    """Return the current speech-recognition availability status."""
    return {
        "enabled":     SPEECH_ENABLED,
        "error":       _IMPORT_ERROR if not SPEECH_ENABLED else "",
        "description": (
            "Speech recognition is available. Click the microphone button to speak."
            if SPEECH_ENABLED
            else f"Speech recognition is not available ({_IMPORT_ERROR or 'SpeechRecognition not installed'}). "
                 "Install with: pip install SpeechRecognition pyaudio"
        ),
    }


def transcribe_from_microphone(
    language: str = "en-ZA",
    timeout: int = 5,
    phrase_time_limit: int = 15,
) -> dict:
    """
    Listen from the system microphone and return a transcription.

    Parameters
    ----------
    language          : BCP-47 language tag (default en-ZA for South African English)
    timeout           : seconds to wait for speech to start
    phrase_time_limit : maximum seconds to record

    Returns
    -------
    dict with keys:
        'success'      : bool
        'text'         : str — transcribed text (empty on failure)
        'error'        : str — error message (empty on success)
    """
    if not SPEECH_ENABLED:
        return {
            "success": False,
            "text":    "",
            "error":   "SpeechRecognition library is not installed.",
        }

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        text = recognizer.recognize_google(audio, language=language)
        return {"success": True, "text": text, "error": ""}

    except sr.WaitTimeoutError:
        return {"success": False, "text": "", "error": "No speech detected. Please try again."}
    except sr.UnknownValueError:
        return {"success": False, "text": "", "error": "Could not understand the audio. Please speak clearly."}
    except sr.RequestError as e:
        return {"success": False, "text": "", "error": f"Speech recognition service error: {e}"}
    except OSError as e:
        return {"success": False, "text": "", "error": f"Microphone error: {e}. Check that a microphone is connected."}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Unexpected error: {e}"}


def transcribe_from_audio_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    language: str = "en-ZA",
) -> dict:
    """
    Transcribe audio from raw bytes (e.g., from a Streamlit audio recorder widget).

    Parameters
    ----------
    audio_bytes  : raw audio bytes (WAV format)
    sample_rate  : audio sample rate in Hz
    language     : BCP-47 language tag

    Returns
    -------
    dict — same structure as transcribe_from_microphone()
    """
    if not SPEECH_ENABLED:
        return {"success": False, "text": "", "error": "SpeechRecognition library is not installed."}

    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        return {"success": True, "text": text, "error": ""}
    except sr.UnknownValueError:
        return {"success": False, "text": "", "error": "Could not understand the audio."}
    except sr.RequestError as e:
        return {"success": False, "text": "", "error": f"Service error: {e}"}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Error processing audio: {e}"}


if __name__ == "__main__":
    status = get_status()
    print(f"Speech recognition enabled : {status['enabled']}")
    print(f"Status                     : {status['description']}")

    if status["enabled"]:
        print("\nListening… (speak now)")
        result = transcribe_from_microphone()
        if result["success"]:
            print(f"Transcribed: {result['text']}")
        else:
            print(f"Error: {result['error']}")

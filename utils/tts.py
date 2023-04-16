import IPython
from gtts import gTTS
from google.cloud import texttospeech
import tempfile
import os

# Set up the text to synthesize
text = "Hello, I am your voice assistant."

# gTTS
def synthesize_gtts(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        fp.seek(0)
        audio_data = fp.read()
    return audio_data

# Google Cloud Text-to-Speech
def synthesize_google_tts(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content

# Synthesize and play the audio
gtts_audio = synthesize_gtts(text)
print("gTTS:")
IPython.display.display(IPython.display.Audio(gtts_audio, format="mp3"))

google_tts_audio = synthesize_google_tts(text)
print("Google Cloud Text-to-Speech:")
IPython.display.display(IPython.display.Audio(google_tts_audio, format="mp3"))

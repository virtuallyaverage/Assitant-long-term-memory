import pyaudio
from pvporcupine import Porcupine
from pvrhino import Rhino
from pvleopard import Leopard
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

class SpeechProcessing:
    def __init__(self, access_key, keyword_path, context_path, porcupine_sensitivity, rhino_sensitivity):
        self.leopard = Leopard(access_key)

        self.model_manager = ModelManager()
        self.tts_model, self.tts_config = self.model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
        self.tts_synthesizer = Synthesizer(self.tts_model, self.tts_config)

        def wake_word_callback():
            pass
        
        self.porcupine = Porcupine(keyword_path=keyword_path, sensitivity=porcupine_sensitivity)
        self.rhino = Rhino(context_path=context_path, sensitivity=rhino_sensitivity)

    def convert_speech_to_text(self, audio_input):
        try:
            text_output = self.leopard.transcribe(audio_input)
            return text_output
        except Exception as e:
            print(f"Error in convert_speech_to_text: {e}")
            return ""

    def convert_text_to_speech(self, text_input):
        try:
            audio_output = self.tts_synthesizer.tts(text_input)
            return audio_output
        except Exception as e:
            print(f"Error in convert_text_to_speech: {e}")
            return None

    def delete(self):
        self.porcupine.delete()
        self.rhino.delete()
        self.leopard.delete()


import wave
import time

if __name__ == "__main__":
    access_key = "YOUR_ACCESS_KEY"
    keyword_path = "data\jarvis_en_windows_v2_2_0.ppn"
    context_path = "data\something-new_en_windows_v2_2_0.rhn"
    audio_input = "data\\test_sound.wav"
    porcupine_sensitivity = 0.5
    rhino_sensitivity = 0.5

    speech_processing = SpeechProcessing(access_key, keyword_path, context_path, porcupine_sensitivity, rhino_sensitivity)

    # Test speech-to-text using Leopard
    text_output = speech_processing.convert_speech_to_text(audio_input)
    print(f"Speech-to-text output: {text_output}")

    # Test text-to-speech using Coqui TTS
    text_input = "Hello, this is a test message."
    audio_output = speech_processing.convert_text_to_speech(text_input)

    # Save the generated audio to a WAV file
    with wave.open("output.wav", "wb") as wav_file:
        wav_file.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        wav_file.writeframes(audio_output)

    # Test Porcupine and Rhino processing
    # Replace this line with the actual audio frames from your audio input source
    audio_frames = [b'\x00\x00' * 512] * 100

    for frame in audio_frames:
        try:
            result = speech_processing.porcupine.process(frame)
            if result:
                print("Wake word detected")
                # Process audio frames with Rhino after wake word detection
                is_finalized = speech_processing.rhino.process(frame)
                if is_finalized:
                    inference = speech_processing.rhino.get_inference()
                    print(f"Intent: {inference['intent']}, Slots: {inference['slots']}")

        except Exception as e:
            print(f"Error processing audio frame: {e}")
            break

    speech_processing.delete()

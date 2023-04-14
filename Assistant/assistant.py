import openai
import os
from Assistant.memory import MilvusAssistant
from typing import List
import pyaudio
import struct
import pvporcupine
import speech_recognition as sr
import webrtcvad
import whisper
import ffmpeg
from time import process_time, sleep
from numpy import frombuffer, int16
from torch import tensor

openai.api_key = os.environ["OPENAI_API_KEY"]



class ChatGPTAssistant(MilvusAssistant):
    def __init__(self, milvus_host: str, milvus_port: str, tokenizer_model_name: str, keys:dict, chat_gpt_model: str = "gpt-3.5-turbo"):
        super().__init__(milvus_host, milvus_port, tokenizer_model_name)
        self.chat_gpt_model = chat_gpt_model
        self.vad = webrtcvad.Vad(2)  # Set aggressiveness level to 2 (default)
        self.max_silence = 500 #ticks
        self.cutoff_after_speak = 60
        self.whisper_model = 'small.en'
        self.porcupine_key = keys['porcupine']
        self.openai_key = keys['openai_api']
        self.skip_list = ["", ".", "exit", "stop", "thanks", "thank you"]
        
        
        
        self.whisper = whisper.load_model(self.whisper_model)

    def generate_response(self, conversation: List[str], max_tokens: int = 150) -> str:
        chat_history = [{'role': 'Helpful AI voice assistant system', 'content': ()}]  # Initialize the system message
        for i, message in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            chat_history.append({"role": role, "content": message})

        prompt = {
            "messages": chat_history,
            "max_tokens": max_tokens
        }
        
        print(f"sending prompt: {prompt}")

        response = openai.Completion.create(
            engine=self.chat_gpt_model,
            prompt=prompt,
            n=1,
            stop=None,
            temperature=0.8,
        )

        return response.choices[0].text.strip()

    def get_relevant_response(self, query: str, top_k: int = 5) -> str:
        relevant_ids = self.get_relevant_info(query, top_k)
        relevant_conversations = [self.collection.load(id) for id in relevant_ids]
        for conversation in relevant_conversations:
            response = self.generate_response(conversation)
            if response:
                return response

        return ""
    
    def wait_for_wake_word(self, wake_words: List[str]):
        porcupine = None
        pa = None
        audio_stream = None

        try:
            print(f"Listening for wake word {wake_words}")
            porcupine = pvporcupine.create(self.porcupine_key, keywords=wake_words)

            pa = pyaudio.PyAudio()

            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length)

            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print("Wake word detected")
                    break
        finally:
            if audio_stream:
                audio_stream.close()
            if pa:
                pa.terminate()
            if porcupine:
                porcupine.delete()

    def record_and_translate(self) -> str:
        with sr.Microphone() as source:
            sleep(.5) #wait for it to stop recording wakeword
            print("recording...")

            silence_counter = 0
            has_spoken = False
            audio_bytes = bytearray()  # Store audio data as bytes
            speaking = False
            while True:
                print(f"silence: {silence_counter}, has spoken: {has_spoken}, speaking: {speaking}")
                
                frame = source.stream.read(320)
                
                if not speaking and has_spoken:
                    silence_counter += 1
                    if silence_counter >= self.cutoff_after_speak:
                        break
                    
                elif not speaking and not has_spoken:
                    silence_counter += 1
                    if silence_counter >= self.max_silence:
                        break
                    
                elif speaking:
                    silence_counter = 0
                    has_spoken = True
                audio_bytes.extend(frame)  # Concatenate audio data as bytes
                sleep(.01) #keep it from taking off
                speaking = self.vad.is_speech(frame, 16000)

            # Convert the byte array to numpy array for whisper
            audio_np = frombuffer(audio_bytes, dtype=int16)
            audio_np = audio_np.astype(float)
            audio_tensor = tensor(audio_np).float()
            
            try:
                text = self.whisper.transcribe(audio_tensor)
                return text["text"]
            except Exception as e:
                print(f"Error converting to text: {e}")
                exit()
            

    def transcribe_audio_whisper(self, audio_data):
        # Save the audio data to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.get_wav_data())

        # Convert the audio to the format required by Whisper
        input_audio = ffmpeg.input("temp_audio.wav")
        output_audio = ffmpeg.output(input_audio, "temp_audio_16k.wav", ar=16000, ac=1)
        ffmpeg.run(output_audio)

        # Load the converted audio and transcribe it using Whisper
        with open("temp_audio_16k.wav", "rb") as f:
            audio_bytes = f.read()

        transcript = whisper.transcribe(audio_bytes)
        return transcript


    def run(self, wake_words: List[str]):
        """_summary_ Continuously waits for a wake word, records and translates the user's query, and generates a response using the existing get_relevant_response method.

        Args:
            wake_words (List[str]): which wakewords to use for picovoice
        """
        
        while True:
            self.wait_for_wake_word(wake_words)
            query = self.record_and_translate()
            print(f"User said: {query}")
            if query.lower() not in self.skip_list:
                response = self.get_relevant_response(query)
                print(f"Assistant: {response}")
                # You can use TTS to speak the response here
                
            else:
                print("skipping")
import openai
import os
from Assistant.memory import ConversationStorage
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
from torch import tensor, cuda



class ChatGPTAssistant(ConversationStorage):
    def __init__(self, tokenizer_model_name: str, keys:dict, chat_gpt_model: str = "gpt-3.5-turbo"):
        super().__init__(tokenizer_model_name)
        self.chat_gpt_model = chat_gpt_model
        self.vad = webrtcvad.Vad(2)  # Set aggressiveness level to 2 (default)
        self.max_silence = 120 #ticks
        self.cutoff_after_speak = 60
        self.whisper_model = 'small.en'
        self.porcupine_key = keys['porcupine']
        print("set openai key to:", keys["openai_api"])
        openai.api_key = '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'
        print("key is :", openai.api_key)
        self.skip_list = ["", ".", "exit", "stop", "thanks", "thank you"]
        self.whisper_device = 'cuda:0' if cuda.is_available() else 'cpu'
        
        
        print(f"Seech to text running on {self.whisper_device}")
        self.whisper = whisper.load_model("medium")
        self.whisper.to(self.whisper_device)

    def generate_response(self, conversation: List[str], max_tokens: int = 1000) -> str:
        def format_prompt(messages):
            formatted_messages = []
            for message in messages:
                if message["role"] == "system":
                    formatted_messages.append(f'System: {message["content"]}')
                elif message["role"] == "user":
                    formatted_messages.append(f'User: {message["content"]}')
                else:
                    formatted_messages.append(f'Assistant: {message["content"]}')
            return "\n".join(formatted_messages)

        chat_history = [{"role": "system", "content": "You are a helpful assistant. listen to the system to get context from past conversations."}]  # Initialize the system message
        for i, message in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            chat_history.append({"role": role, "content": message})

        relevant_conversations = self.search_conversations(":".join(conversation))  
        chat_history.append({"role":"system", "content":"Relevent conversations from the past:"})
        for old_conversation in relevant_conversations:
            chat_history.append(old_conversation)
        
        print(f"sending off: {chat_history}")
        prompt = {
            "messages": chat_history,
            "max_tokens": max_tokens
        }
        
        
        print(f"sending prompt: {prompt}")

        formatted_prompt = format_prompt(prompt["messages"])
        
        response = openai.ChatCompletion.create(
            engine=self.chat_gpt_model,
            prompt=prompt,
            n=1,
            max_tokens=max_tokens,
            stop=None,
            temperature=0.8,
        )


        return response.choices[0].text.strip()
    
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


    def run(self, wake_words: List[str]=["computer", "jarvis"]):
        """_summary_ Continuously waits for a wake word, records and translates the user's query, and generates a response using the existing get_relevant_response method.

        Args:
            wake_words (List[str]): which wakewords to use for picovoice
        """
        
        while True:
            self.wait_for_wake_word(wake_words)
            query = self.record_and_translate()
            print(f"User said: {query}")
            
            if query.lower() not in self.skip_list:
                response = self.generate_response([query])
                print(f"Assistant: {response}")
                # You can use TTS to speak the response here
                
            else:
                print("skipping")
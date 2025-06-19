import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import ollama
import wave
from whisper_cpp_python import Whisper
from piper import PiperVoice

def record_audio(filename="input.wav", duration=2, sample_rate=16000, device_index=None):
    """
    Records audio from the microphone and saves it to a WAV file.
    You can specify a device_index; otherwise, it uses the system's default input device.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_index)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, sample_rate, recording)  # Save as WAV file
    print(f"Recording finished and saved to {filename}")
    return filename

def speech_to_text(audio_file):
    """
    Transcribes audio to text using Whisper.cpp.
    """
    try:
        whisper = Whisper(model_path="ggml-base.en.bin")
    except Exception as e:
        print(f"Error initializing Whisper: {e}")
        print("Please make sure you have downloaded a whisper model (e.g., `ggml-base.en.bin`)")
        return None

    print("Transcribing audio...")
    result = whisper.transcribe(audio_file)
    text = result["text"]
    print(f"Transcription: {text}")
    return text

def get_llm_response(text):
    """
    Gets a response from the Ollama model using TinyLlama.
    """
    print("Getting response from Ollama...")
    try:
        response = ollama.chat(
            model='tinyllama',
            messages=[{'role': 'user', 'content': text}]
        )
        llm_response = response['message']['content']
        print(f"Ollama response: {llm_response}")
        return llm_response
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        print("Please ensure Ollama is running and you have pulled the tinyllama model (e.g., `ollama pull tinyllama`).")
        return "I am having trouble connecting to my brain."


def text_to_speech(text, model_path="en_US-joe-medium.onnx", output_file="output.av"):
    """
    Converts text to speech using Piper TTS.
    """
    if not text:
        print("No text to synthesize.")
        return

    if not os.path.exists(model_path):
        print(f"Piper TTS model not found at {model_path}")
        print("Please download a voice model from https://huggingface.co/rhasspy/piper-voices/tree/main")
        return

    print("Synthesizing speech...")
    voice = PiperVoice.load(model_path)
    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    
    print(f"Speech synthesized and saved to {output_file}")

    try:
        os.system(f"aplay {output_file}")
    except Exception as e:
        print(f"Could not play audio: {e}. You can play '{output_file}' manually.")


def main():
    """
    Main function to run the voice assistant loop.
    """
    print("Starting voice assistant. Press Ctrl+C to exit.")
    try:
        while True:
            input("Press Enter to start recording...")
            audio_file = record_audio()
            user_text = speech_to_text(audio_file)

            if user_text:
                llm_response = get_llm_response(user_text)
                if llm_response:
                    text_to_speech(llm_response)

    except KeyboardInterrupt:
        print("\nExiting voice assistant.")

if __name__ == "__main__":
    main() 

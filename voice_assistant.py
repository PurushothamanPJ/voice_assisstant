import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import ollama
import wave
import threading
import queue
import time
from whisper_cpp_python import Whisper
from piper import PiperVoice
import tempfile

class OptimizedVoiceAssistant:
    def __init__(self, whisper_model="ggml-base.en.bin", piper_model="en_US-joe-medium.onnx"):
        """Initialize the voice assistant with pre-loaded models."""
        print("Initializing voice assistant...")
        
        # Pre-load Whisper model
        try:
            print("Loading Whisper model...")
            self.whisper = Whisper(model_path=whisper_model)
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Please make sure you have downloaded a whisper model (e.g., `ggml-base.en.bin`)")
            self.whisper = None
        
        # Pre-load Piper voice model
        if os.path.exists(piper_model):
            print("Loading Piper voice model...")
            self.voice = PiperVoice.load(piper_model)
            print("Piper voice model loaded successfully")
        else:
            print(f"Piper model not found at {piper_model}")
            print("Please download a voice model from https://huggingface.co/rhasspy/piper-voices/tree/main")
            self.voice = None
        
        # Audio processing settings
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'int16'
        
        # Threading for parallel processing
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        
        print("Voice assistant initialized successfully!")

    def record_audio(self, filename=None, duration=5, device_index=None):
        """
        Records audio from the microphone and saves it to a temporary WAV file.
        Optimized version with better buffer management.
        """
        if filename is None:
            # Use a temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            filename = temp_file.name
            temp_file.close()
        print("Recording...")
        try:
            recording = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=self.channels, 
                dtype=self.dtype, 
                device=device_index
            )
            sd.wait()  # Wait until recording is finished
            wav.write(filename, self.sample_rate, recording)
            print(f"Recording finished and saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def speech_to_text(self, audio_file):
        """
        Optimized speech-to-text using Whisper with faster inference parameters.
        """
        if not self.whisper:
            print("Whisper model not loaded")
            return None
        print("Transcribing audio...")
        start_time = time.time()
        try:
            # Remove unsupported parameters for your Whisper version
            result = self.whisper.transcribe(audio_file)
            text = result["text"].strip()
            # Clean up audio file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                print(f"Error deleting audio file: {e}")
            transcribe_time = time.time() - start_time
            print(f"Transcription completed in {transcribe_time:.2f}s: {text}")
            return text
        except Exception as e:
            print(f"Error during transcription: {e}")
            # Clean up audio file even on error
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e2:
                print(f"Error deleting audio file after error: {e2}")
            return None

    def get_llm_response(self, text):
        """
        Gets a response from the Ollama model with optimized settings.
        """
        if not text:
            return None
            
        print("Getting response from Ollama...")
        start_time = time.time()
        
        try:
            # Optimized Ollama call with timeout and response limits
            response = ollama.chat(
                model='tinyllama',
                messages=[{'role': 'user', 'content': text}],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 150,  # Limit response length for speed
                    'num_predict': 150,  # Alternative parameter name
                }
            )
            
            llm_response = response['message']['content']
            response_time = time.time() - start_time
            print(f"Ollama response received in {response_time:.2f}s: {llm_response}")
            
            # Extract only the first assistant reply (clean up format)
            if "Assistant:" in llm_response:
                parts = llm_response.split("Assistant:")
                if len(parts) > 1:
                    first_reply = parts[1].split("User:")[0].strip()
                    return first_reply
            
            # Return cleaned response
            return llm_response.strip()
            
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            print("Please ensure Ollama is running and you have pulled the tinyllama model.")
            return "I am having trouble connecting to my brain."

    def text_to_speech(self, text, output_file=None):
        """
        Converts text to speech using Piper TTS with optimizations.
        """
        if not text or not self.voice:
            if not text:
                print("No text to synthesize.")
            if not self.voice:
                print("Piper voice model not loaded.")
            return None
        
        if output_file is None:
            output_file = f"output_{int(time.time())}.wav"
        
        print("Synthesizing speech...")
        start_time = time.time()
        
        try:
            # Synthesize speech
            with wave.open(output_file, "wb") as wav_file:
                self.voice.synthesize(text, wav_file)
            
            synthesis_time = time.time() - start_time
            print(f"Speech synthesized in {synthesis_time:.2f}s and saved to {output_file}")
            
            # Play audio asynchronously
            self._play_audio_async(output_file)
            return output_file
            
        except Exception as e:
            print(f"Error during text-to-speech: {e}")
            return None

    def _play_audio_async(self, audio_file):
        """Play audio in a separate thread to avoid blocking."""
        def play():
            try:
                print(f"Playing audio: {audio_file}")
                os.system(f"aplay {audio_file} 2>/dev/null")
                # Clean up after playing
                time.sleep(0.5)  # Small delay to ensure file is not in use
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                print(f"Could not play audio: {e}. You can play '{audio_file}' manually.")
        
        threading.Thread(target=play, daemon=True).start()

    def process_audio_worker(self):
        """Worker thread for processing audio to text."""
        while not self.shutdown_event.is_set():
            try:
                audio_file = self.audio_queue.get(timeout=1)
                if audio_file is None:  # Shutdown signal
                    break
                    
                text = self.speech_to_text(audio_file)
                if text:
                    self.text_queue.put(text)
                    
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio worker: {e}")

    def process_llm_worker(self):
        """Worker thread for processing text through LLM."""
        while not self.shutdown_event.is_set():
            try:
                text = self.text_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                    
                response = self.get_llm_response(text)
                if response:
                    self.tts_queue.put(response)
                    
                self.text_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in LLM worker: {e}")

    def process_tts_worker(self):
        """Worker thread for text-to-speech processing."""
        while not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                    
                self.text_to_speech(text)
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in TTS worker: {e}")

    def start_worker_threads(self):
        """Start all worker threads for parallel processing."""
        self.threads = [
            threading.Thread(target=self.process_audio_worker, daemon=True),
            threading.Thread(target=self.process_llm_worker, daemon=True),
            threading.Thread(target=self.process_tts_worker, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        print("Worker threads started for parallel processing")
        return self.threads

    def stop_worker_threads(self):
        """Stop all worker threads gracefully."""
        print("Stopping worker threads...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Send shutdown signals to queues
        self.audio_queue.put(None)
        self.text_queue.put(None)
        self.tts_queue.put(None)
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)

    def run_threaded(self):
        """
        Main function with full threading optimization.
        Best performance but more complex.
        """
        print("Starting optimized voice assistant with threading. Press Ctrl+C to exit.")
        
        # Start worker threads
        self.start_worker_threads()
        
        try:
            while True:
                input("Press Enter to start recording...")
                
                total_start = time.time()
                
                # Record audio
                audio_file = self.record_audio(duration=5)
                if audio_file:
                    # Add to processing queue
                    self.audio_queue.put(audio_file)
                    
                    # Show queue status
                    print(f"Queue status - Audio: {self.audio_queue.qsize()}, "
                          f"Text: {self.text_queue.qsize()}, TTS: {self.tts_queue.qsize()}")
                
                total_time = time.time() - total_start
                print(f"Recording and queuing completed in {total_time:.2f}s")
                
        except KeyboardInterrupt:
            print("\nShutting down voice assistant...")
            self.stop_worker_threads()

    def run_simple(self):
        """
        Simple optimized version without threading.
        Easier to debug, still much faster than original.
        """
        print("Starting simple optimized voice assistant. Press Ctrl+C to exit.")
        
        try:
            while True:
                input("Press Enter to start recording...")
                
                total_start = time.time()
                
                # Sequential processing with optimizations
                audio_file = self.record_audio(duration=5)
                if not audio_file:
                    continue
                
                user_text = self.speech_to_text(audio_file)
                if not user_text:
                    continue
                
                llm_response = self.get_llm_response(user_text)
                if llm_response:
                    self.text_to_speech(llm_response)
                
                total_time = time.time() - total_start
                print(f"Total pipeline completed in {total_time:.2f}s")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nExiting voice assistant.")

    def run_original_style(self):
        """
        Keep the original function style but with optimizations.
        Drop-in replacement for your original main() function.
        """
        print("Starting voice assistant (original style with optimizations). Press Ctrl+C to exit.")
        
        try:
            while True:
                input("Press Enter to start recording...")
                audio_file = self.record_audio()
                user_text = self.speech_to_text(audio_file)
                if user_text:
                    llm_response = self.get_llm_response(user_text)
                    if llm_response:
                        self.text_to_speech(llm_response)
        except KeyboardInterrupt:
            print("\nExiting voice assistant.")

# Original function style (optimized versions)
def record_audio(filename="input.wav", duration=5, sample_rate=16000, device_index=None):
    """Original function - kept for compatibility."""
    assistant = OptimizedVoiceAssistant()
    return assistant.record_audio(filename, duration, device_index)

def speech_to_text(audio_file):
    """Original function - kept for compatibility."""
    assistant = OptimizedVoiceAssistant()
    return assistant.speech_to_text(audio_file)

def get_llm_response(text):
    """Original function - kept for compatibility."""
    assistant = OptimizedVoiceAssistant()
    return assistant.get_llm_response(text)

def text_to_speech(text, model_path="en_US-joe-medium.onnx", output_file="output.wav"):
    """Original function - kept for compatibility."""
    assistant = OptimizedVoiceAssistant(piper_model=model_path)
    return assistant.text_to_speech(text, output_file)

def main():
    """
    Main function: always use the fastest threaded mode, no prompt.
    """
    assistant = OptimizedVoiceAssistant()
    assistant.run_threaded()

if __name__ == "__main__":
    main()

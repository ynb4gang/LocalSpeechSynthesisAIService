import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import SpeechSynthesisService

console = Console()
stt = whisper.load_model("base.en")
tts = SpeechSynthesisService()

template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(),
)

def capture_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): Event to signal stopping of audio capture.
        data_queue (queue.Queue): Queue to store recorded audio data.
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe_audio(audio_np: np.ndarray) -> str:
    """
    Transcribes audio data using Whisper speech recognition.

    Args:
        audio_np (numpy.ndarray): Audio data for transcription.

    Returns:
        str: Transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)
    text = result["text"].strip()
    return text

def get_ai_response(text: str) -> str:
    """
    Generates a response using the Llama-2 language model.

    Args:
        text (str): Input text for response generation.

    Returns:
        str: AI-generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response

def play_generated_audio(sample_rate, audio_array):
    """
    Plays audio data with the given sample rate.

    Args:
        sample_rate (int): Sample rate for audio.
        audio_array (numpy.ndarray): Audio array to play.
    """
    sd.play(audio_array, sample_rate)
    sd.wait()

if __name__ == "__main__":
    console.print("[cyan]Your's AI assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input("Let's just talk!\nPress Enter to start recording, then press Enter again to stop. ")

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=capture_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe_audio(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_ai_response(text)
                    sample_rate, audio_array = tts.generate_long_audio(response)

                console.print(f"[cyan]Assistant: {response}")
                play_generated_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
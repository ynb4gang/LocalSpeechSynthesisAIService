
import nltk
import torch
import warnings
import numpy as np
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

class SpeechSynthesisService:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the SpeechSynthesisService class.

        Args:
            device (str, optional): Device to use for the model, either "cuda" for GPU or "cpu". Defaults to GPU if available.
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        self.model.to(self.device)

    def generate_audio(self, text: str, voice_setting: str = "v2/en_speaker_1"):
        """
        Generates audio from text using a specified voice setting.

        Args:
            text (str): Text to synthesize.
            voice_setting (str, optional): Voice setting for synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: A tuple with the sample rate and generated audio array.
        """
        inputs = self.processor(text, voice_preset=voice_setting, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def generate_long_audio(self, text: str, voice_setting: str = "v2/en_speaker_1"):
        """
        Generates audio for longer text by splitting it into sentences.

        Args:
            text (str): Long-form text to synthesize.
            voice_setting (str, optional): Voice setting for synthesis. Defaults to "v2/en_speaker_1".

        Returns:
            tuple: Sample rate and concatenated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sentence in sentences:
            sample_rate, audio_array = self.generate_audio(sentence, voice_setting)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)
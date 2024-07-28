import os
from fastai.vision.all import *
import numpy as np
import gradio as gr
import librosa
import matplotlib.pyplot as plt

__all__ = ['learner', 'labels', 'interface', 'detect']

tmp_filename = 'temp.png'
learner = load_learner('export.pkl')
labels = learner.dls.vocab

"""
The following function is used to chunk the audio into chunk_duration
seconds long blocks with overlap equal to overlap_duration seconds.
"""
def chunk_audio(audio, sample_rate, chunk_duration, overlap_duration):
    chunks = []
    chunk_length = chunk_duration * sample_rate
    overlap_length = overlap_duration * sample_rate

    start = 0
    while start + chunk_length <= len(audio):
        end = start + chunk_length
        chunks.append(audio[start:end])
        start += (chunk_length - overlap_length)
    # Handle the last chunk if it's smaller than chunk_length
    if start < len(audio):
        chunks.append(audio[-chunk_length:])
    return chunks

def audio_to_mel_spectrogram(audio, sample_rate, dest_path):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # resizing figure to 224x224px images since the model was trained on that size
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(spectrogram_db, sr=sample_rate, hop_length=512)
    plt.tight_layout()
    plt.savefig(dest_path, bbox_inches='tight', pad_inches=0)

def process(audio_file):
    try:
        audio, sample_rate = librosa.load(audio_file)
        chunks_10_sec = chunk_audio(audio, sample_rate, 10, 5)
        results = []
        for chunk in chunks_10_sec:
            audio_to_mel_spectrogram(chunk, sample_rate, tmp_filename)
            _, _, probs = learner.predict(tmp_filename)
            results.append(probs)
            os.remove(tmp_filename)
        final_result = np.mean(results, axis=0)
        return dict(zip(labels, map(float, final_result)))
    except Exception as e:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


demo = gr.Interface(
    fn=process,
    inputs=gr.Audio(sources=['upload'], type='filepath'),
    outputs=gr.Label(),
    title="What genre is it?",
)

demo.launch()
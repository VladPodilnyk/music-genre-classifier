---
title: Music genre classifier
emoji: 🦀
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.32.2
app_file: app.py
pinned: false
sse: true
---

### Music genre classifier using DL

The project showcases an example of audio processing using computer vision. In particular it uses a family of image recognition neural networks
to identify patterns/features of a [MEL spectrogram](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) and use this "knowledge" to derive a music genre of a song.

The repo contains a notebook that shows the process of data exploration and model training, together
with a small script featuring [Gradio app](https://huggingface.co/spaces/podil97/music_genre_classifier) build on top of the trained model.

#### Training process 🏋🏻‍♂️

The [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset contains short audio samples for different musing genres.
Before the starting training, those samples has to be transformed into images. Using `librosa` package audio samples are transformed into MEL spectrograms that describe how "frequency" component of a song changes over time.

`fastai` package is used for training. Training and validation sets have 80% and 20% split.
During training all images are resized to fit `resnet50` inputs.

#### A few notes about the app 📓

The app uses a trained model to "predict" an appropriate label for a song.
There are a few things that have to be mentioned. First, most audio tracks uploaded by users are not samples meaning the duration of a track is 2-3min. That's way larger than an average duration of a sample used for training (20-30sec).
To mitigate this issue, behind the scenes, an uploaded song is cut into small chunks with some overlap region. The prediction is made for all chunks and then the average is taken to make a final verdict.

Another thing to remember is that each chunk has to be converted into MEL spectrogram and resized to fit the model inputs (in this case it's 224x224px images).

#### Running locally 👩‍💻

In case you want to play with this project locally, please make sure that your dataset
is located under the following path `dataset/Data/genres_original/`.

Also, the repo is not containerized hence you have to ensure that your python env contains
all the necessary dependencies like `torch`, `fastai`, `fastbook`, `librosa` and others.
My personal recommendation to install `miniforge` and use `mamba` to setup your env.

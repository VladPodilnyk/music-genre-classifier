### Music genre classification using deep learning

Super simple music genre classifier using pre-trained resnet50 model.
The idea is pretty simple, take audio files from the famous [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset, covert them into Mel spectrograms
and use CNNs to recognize patterns and differentiate between different music genres.

#### Important notes:

The model isn't very accurate at the moment. The reason for that, I believe, is that the input
for ResNet family models is 224x224 pixels image wheres in reality we might want much larger pictures
to be able to recognize unique traits of each spectrogram.
Another thing to keep in mind is that training data contains 10 sec long audio files, but for the app
it's expected that a user drops a song of an arbitrary length. To cover such case I decided to split audio file
in chunks of 10 sec with some small overlap window. When all chunks are processed, the final result will be equal
to the average of partial results per chunk.

__Things to improve__:

- Now, since I believe the input size is one of the major obstacles, I think it's worth to try to come up
  with a custom model architecture.
- Try different approach (without image processing)
- Long audio tracks handling


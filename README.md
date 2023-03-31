# Speech-Emotion-Conversion-with-PyTorch

#### Abstract: This project intends to develop a high-quality many-to-many emotion conversion model that can convert a speech audio into specified emotions. To achieve this goal, we use pretrained parallel CNN-transformers model with high speech  emotion recognition accuracy to extract emotion embeddings

## Dependencies

```
librosa~=0.9.2
pandas~=1.5.3
numpy~=1.23.2
torchinfo~=1.7.2
pyworld~=0.3.2
scikit-learn~=1.2.1
```

Install by `pip install -r requirements.txt`

## Project Architecture

```
.
├── MCD.py
├── ParallelCNNTrans.py
├── README.md
├── VAE_Trainer.py
├── VAW_GAN.py
├── VAW_Trainer.py
├── analyzer.py
├── audio_speech_actors_01-24
├── bin
├── build.py
├── convert.py
├── converted
├── emotion_representation.py
├── logdir
├── main.py
├── model
├── notebooks
│   ├── WorldVocoder.ipynb
│   ├── parallel_cnn_transformer.ipynb
│   └── visualization.ipynb
├── preprocess_utils.py
├── ravdess_complete_feature_embedding.npy
├── requirements.txt
├── trainer.py
└── util.py
```

### Stage I: Speech Emotion Recognition

Access Parallel CNN-Transformer Model Architecure from `ParallelCNNTrans.py`. Extract emotion embedding through: `emotion_representation.py`.

The training process and training pipeline for SER model is described in `notebook/parallel_cnn_transformer.ipynb`. The pretrained weight for SER model is saved in `./bin/models/`

### Stage II: Emotion Conversion

To preprocess RAVDESS dataset and linking embedding with processed data, we use `build.py` to build the pre-processed dataset with normalization. The extracted compelete preprocessed data with embedding is saved in `./ravdess_complete_feature_embedding.npy`, and the normalization data is stored in `./bin/etc/`

Our VAW Model Architecture is showed in `VAW_GAN.py`.

To train VAE, we can call `VAE_Trainer.py`,the training status will be logged in logdirs and the final model will be saved in `./model`. To train the VAW, we need to call `VAW_Trainer.py`and specified the loading VAE weights dir (where the VAE be saved). Otherwise we can call `main.py` with `Trainer.py` to complete VAE and VAW all at once.

Some of our visualization script is showed in the `./notebook/visualization.ipynb`, be catious about the type of logging data, please use `def process()` to process the loading data.

### Stage III: Audio Generator:

Call `Convert.py`, specify the path for loading weight and the source and target, the generated audio will be saved in `./converted/`.

To evaluate the result, call MCD.py to compute the MCD value(s).

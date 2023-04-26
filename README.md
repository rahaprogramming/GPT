# GPT
generative pre-trained transformer for spoken text

The first step is to collect speech data. There are many ways to do this, but one common approach is to use a public dataset like the Mozilla Common Voice dataset. This dataset consists of thousands of hours of speech recordings in many different languages. The steps to run this program is as follows:

1. collect speech data using SpeechData.py
2. preprocess it to prepare it for training the Wav2Vec2 model using PreProcessSpeechData.py
3. Train the Wav2Vec2 Model using TrainModel.py
4. Now that we have trained the Wav2Vec2 model, we can use it to generate text from speech with GenerateSpeech.py

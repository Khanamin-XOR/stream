import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
import librosa
import cv2
from keras.applications.vgg16 import preprocess_input
from librosa.util import fix_length
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import *
from keras import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.image import *
from keras.utils.vis_utils import *
from sklearn.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow_addons as tfa
import subprocess
if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/Khanamin-XOR/stream/main/final_model.h5"'], shell=True)
model = load_model(model.h5)
st.title("Emotion Classification with Image and Audio with Multimodal Approch")
st.header("Example of emotion classification")
st.text("Upload an  image and a Audio file for classification as Happy, Neural, Anger and Disgusted")
uploaded_file_1 = st.file_uploader("Choose a Image ...", type="jpg")
uploaded_file_2 = st.file_uploader("Choose a Audio ...", type="wav")
if uploaded_file_1 and uploaded_file_2 is not None:
	image = Image.open(uploaded_file_1) #Load Image
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	image_array = np.asarray(image)
	gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x, y, w, h) in faces:
		cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 0, 255), 2)
		faces = image_array[y:y + h, x:x + w]
		cv2.imwrite('face.jpg', faces)

	face_images = Image.open('face.jpg')
	st.image(face_images, caption='Extracted Image.', use_column_width=True)

	def load_wav(x, get_duration=True):
		samples, sample_rate = librosa.load(x, sr=12000)
		if get_duration:
			duration = librosa.get_duration(samples, sample_rate)
			return [samples, duration]
		else:
			return samples

	raw_data = load_wav(uploaded_file_2,get_duration=False)
	audio = uploaded_file_2.read()
	st.audio(audio, format='audio/wav')


	st.write("")
	st.write("Classifying...")
	
	def classification_report(image,audio,weights):
		#model = load_model(weights)
		data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)
		size = (160, 160)
		image = ImageOps.fit(image, size, Image.ANTIALIAS)
		image_array = np.asarray(image)
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
		data[0] = normalized_image_array
		def convert_to_spectrogram(raw_data):
			'''converting to spectrogram'''
			spectrum = librosa.feature.melspectrogram(y=raw_data, sr=sample_rate, n_mels=64)
			logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
			return logmel_spectrum

		def convert_to_mfcc(raw_data):
			mfccs = librosa.feature.mfcc(y=raw_data, sr=sample_rate, n_mfcc=64)
			return mfccs
		
		def zero_crossings_rates(raw_data):
			zero_crossings = librosa.feature.zero_crossing_rate(y=raw_data, frame_length=2048, hop_length=512, center=True)
			return zero_crossings

		def convert_rmse(raw_data):
			rmse = librosa.feature.rms(y=raw_data, frame_length=2048, hop_length=512)
			return rmse

		def convert_Spectral_centriod(raw_data):
			spectral_centroids = librosa.feature.spectral_centroid(y=raw_data, sr=12000,
                                                         n_fft=2048, hop_length=512, freq=None,
                                                         win_length=None, window='hann', center=True,
                                                         pad_mode='constant')
			return spectral_centroids

		sample_rate = 12000
		max_length  = sample_rate*10
		padded_audio = fix_length(audio, size=max_length)
		X_train_spectrogram = convert_to_spectrogram(padded_audio)
		X_train_mfcc = convert_to_mfcc(padded_audio)
		X_train_zcr = zero_crossings_rates(padded_audio)
		X_train_rmse =convert_rmse(padded_audio)
		X_train_SC = convert_Spectral_centriod(padded_audio)
		X_train_new = np.vstack((X_train_spectrogram,X_train_mfcc,X_train_zcr,
                           X_train_rmse, X_train_SC))

		X_train_new = X_train_new.reshape(1,131,235)
		values = [data,X_train_new]
		predictions = model.predict(values)
		return np.argmax(predictions)

	label = classification_report(face_images,raw_data, model)
	if label == 0:
		st.write("The Emotion is Anger")
	elif (label ==1):
		st.write("The Emotion is Disgusted")
	elif (label ==2):
		st.write("The Emotion is Happy")
	else:
		st.write("The Emotion is Neutral")

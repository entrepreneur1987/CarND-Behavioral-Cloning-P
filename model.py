import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import math
import matplotlib.pyplot as plt

base_path = '/Users/luyaoli/self-driving-nd/'
train_data_dirs = ['track1_new', 'track1_new_recovery', 'track1_new_curve']


class Data:
	""" Data stores image path and the corresponding steering angle information """
	def __init__(self, img_path, angle):
		self.img_path = img_path
		self.angle = angle

def get_all_data(target_data_dirs):
	"""
	Iterate through all target directories, dropping partial data points of
	straight roads, convert from CSV line to Data.
	"""
	data = []
	for data_dir in target_data_dirs:
		driving_log = base_path + data_dir + '/driving_log.csv'
		with open(driving_log) as f:
			reader = csv.reader(f)
			next(reader)
			for line in reader:
				angle = float(line[3])
				r = random.random()
				# only keep 20% of the data points from straight roads
				if abs(angle) < 0.01 and r > 0.2:
					continue
				data.append(Data(line[0], angle))
				data.append(Data(line[1], angle + 0.25))
				data.append(Data(line[2], angle - 0.25))
	return data

def _get_hist(angle_array):
	""" Gets histogram array from the angle array, with bin=50 """
	return np.histogram(angle_array, bins=50)


def _plot_angle_distribution(angle_array, idx, sub_idx):
	""" Plots steering angle distribution graph for visualization """
	plt.figure(idx)
	plt.subplot(2, 1, sub_idx)
	plt.hist(angle_array, bins=50, rwidth=0.5)
	plt.xlabel("Steering Angle (radian)")
	plt.ylabel("Number of pictures")
	print(len(angle_array))
	plt.show()


# generates all samples and split into training and validation (20%)
all_samples = get_all_data(train_data_dirs)
train_samples, valid_samples = train_test_split(all_samples, test_size = 0.2)
angle_array = [d.angle for d in all_samples]

# _plot_angle_distribution(angle_array, 1, 1)

def sample_generator(samples, batch_size=128):
	""" Sample generator which store one batch of data in memory at a time """
	num_samples = len(samples)
	while True:
		# shuffle all samples for randomness
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_sample = samples[offset: offset + batch_size]
			images = []
			measurements = []
			for sample in batch_sample:
				img_path = sample.img_path
				img = cv2.imread(img_path)
				# convert BGR to YUV
				img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
				# crop the image, cut the top 70 pix and botton 25 pix
				img = img[70:135, 0:320]
				# resize the image to 66 x 200 to fit into the NVidia model
				img = cv2.resize(img, (200, 66))
				images.append(img)
				measurements.append(sample.angle)

				# flip the image, here comment it out as the dataset collected
				# already big enough, so training is faster, can uncomment when
				# time permits

				# images.append(cv2.flip(img, 1))
				# measurements.append(-sample.angle)
			X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)

batch_size = 128
train_generator = sample_generator(train_samples, batch_size)
valid_generator = sample_generator(valid_samples, batch_size)

model = Sequential()
# normalize the input
model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(66, 200, 3)))

# Convolutional layer with a filter of (5, 5) and depth = 24
# Activation function set to 'RELU', strides = (2,2)
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))

# Convolutional layer with a filter of (5, 5) and depth = 36
# Activation function set to 'RELU', strides = (2,2)
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))

# Convolutional layer with a filter of (5, 5) and depth = 48
# Activation function set to 'RELU', strides = (2,2)
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))

# Convolutional layer with a filter of (3, 3) and depth = 64
# Activation function set to 'RELU'
model.add(Conv2D(64, (3,3), activation='relu'))

# Convolutional layer with a filter of (3, 3) and depth = 64
# Activation function set to 'RELU'
model.add(Conv2D(64, (3,3), activation='relu'))

# Flatten layer
model.add(Flatten())

# Fully-connected layer with output = 1164
model.add(Dense(1164))

# Fully-connected layer with output = 100
model.add(Dense(100))

# Fully-connected layer with output = 50
model.add(Dense(50))

# Fully-connected layer with output = 10
model.add(Dense(10))

# Fully-connected layer with output = 1
model.add(Dense(1))

# compile the model with 'mse' as the loss function and use 'adam' optimizer
model.compile(loss='mse', optimizer='adam')

# train the model with generator method
model.fit_generator(train_generator, len(train_samples) / batch_size, 1, validation_data = valid_generator, validation_steps = len(valid_samples)/batch_size)
model.save('model_local_generator.h5')

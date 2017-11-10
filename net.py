from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import sys

from makedata import make_split_data

# Get the data
X_train, X_test, Y_train, Y_test = make_split_data(2000)


train = False if len(sys.argv) < 2 else (sys.argv[1] == "train")

if train:
	model = Sequential()
	model.add(Dense(3, input_dim=3, activation='relu'))
	model.add(Dense(4, input_dim=3, activation='relu'))
	model.add(Dense(10, input_dim=4, activation='relu'))
	model.add(Dense(4, input_dim=10, activation='relu'))
	model.add(Dense(1, activation='sigmoid', input_dim=4))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, Y_train, epochs=500)
	model.save('./model.h5')
else:
	model = load_model('./model.h5')

scores = model.evaluate(X_test, Y_test)

predict = 'white' if model.predict(np.asarray([[89, 50, 100]])) >= 0.5 else 'black'

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("Prediction", predict)
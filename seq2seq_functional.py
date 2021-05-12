from keras.datasets import mnist  

(X_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

def seq2seq():
    input_ = keras.layers.Input(shape=[28, 28])
    flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)
    hidden1 = keras.layers.Dense(2**14, activation="relu")(flatten)
    hidden2 = keras.layers.Dense(512, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(28*28, activation='relu')(hidden2)
    reshap = keras.layers.Reshape((28, 28))(hidden3)
    concat_ = keras.layers.Concatenate()([input_, reshap])
    flatten2 = keras.layers.Flatten(input_shape=[28, 28])(concat_)
    output = keras.layers.Dense(10, activation='softmax')(flatten2)
    return input_,output

io=seq2seq()
model = keras.Model(inputs=[io[0]], outputs=[io[1]] )
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1]*100)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
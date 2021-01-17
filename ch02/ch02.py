import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
callbacks = [tf.keras.callbacks.TensorBoard("./logs_keras")]
model.fit(
    x_train,
    y_train,
    epochs=25,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)


estimator = tf.keras.estimator.model_to_estimator(model, model_dir="./estimator_dir")

BATCH_SIZE = 32


def train_input_fn():
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()
    return train_dataset


estimator.train(train_input_fn, steps=len(x_train) // BATCH_SIZE)


# create a writer and log information
writer = tf.summary.create_file_writer("./model_logs")
with writer.as_default():
    tf.summary.scalar("custom_log", 10, step=3)

# or

# ex: manually log accuracy
accuracy = tf.keras.metrics.Accuracy()
# in practice this would come from the model
ground_truth, predictions = [1, 0, 1], [1, 0, 0]
accuracy.update_state(ground_truth, predictions)
tf.summary.scalar("accuracy", accuracy.result(), step=4)

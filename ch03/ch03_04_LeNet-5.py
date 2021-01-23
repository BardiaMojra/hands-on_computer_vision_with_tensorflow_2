# ch03.03 LeNet-5: building and training
# we only code feedforward process and tensorflow automatically performs backprop
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv

num_classes = 10
im_rows, im_cols, im_ch = 28, 28, 1
input_shape = (im_rows, im_cols, im_ch)

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalize 8-bit pixel intensity value from 0-255 to 0-1 (normalize)
x_train, x_test = x_train / 255.0, x_test / 255.0
# reshape data to column vectors
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# instantiating conv layers
class SimpleConvLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_kernels=32, kernel_size=(3, 3), strides=(1, 1), use_bias=True
    ):
        """
        Initialize the layer.
        :param num_kernels: num of kernels for convolving
        :param kernel_size: kernel size (H x W)
        :param strides: vertical and horizontal stride as list
        :param use_bias: flag to add a bias after convolution/ before activation
        """
        # first we call the 'Layer' super __init__(), as it initializes the layer
        super().__init__()
        # then assign the default parameters to class data structure
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias

    def build(self, input_shape):
        """
        Build the layer, initializing its parameters/variables. This is called
        internally the first time the layer is used, though it can be manually called.
        :param input_shape: input shape for the layer will receive (e.g. B x H x W x C)
        """

        num_input_channels = input_shape[-1]  # last item on the list
        # set shape of the tensor representing the kernels
        kernels_shape = (*self.kernel_size, num_input_channels, self.num_kernels)

        # for this example use Glorot distribution for init values:
        glorot_uni_initializer = tf.initializers.GlorotUniform()
        self.kernels = self.add_weight(
            name="kernels",
            shape=kernels_shape,
            initializer=glorot_uni_initializer,
            trainable=True,
        )  # make kernel variables trainable

    def call(self, inputs):
        """
        Call the layer and perform its operations.
        :param input: input tensor
        :return: output tensor
        """
        # perform convolution
        z = tf.nn.conv2d(
            inputs, self.kernels, strides=[1, *self.strides, 1], padding="VALID"
        )

        if self.use_bias:  # use bias term if it's not zero
            z = z + self.use_bias

        # use activation function (e.g. ReLU) and return output, z
        return tf.nn.relu(z)

    def get_config(self):
        """
        Helper function to define the layer and its parameters.
        :return: dictionary containing layer's configuration
        """
        return {
            "num_kernels": self.num_kernels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "use_bias": self.use_bias,
        }


# implementing LeNet-5
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class LeNet5(Model):
    def __init__(self, num_classes):
        """
        Initialize the model.
        :param num_classes: number of classes to predict from.
        """
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5, 5), padding="SAME", activation="relu")
        self.conv2 = Conv2D(16, kernel_size=(5, 5), activation="relu")
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation="relu")
        self.dense2 = Dense(84, activation="relu")
        self.dense3 = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        """
        Call the layers as the model would.
        :param inputs: Input tensor.
        :return: Output tensor.
        """
        x = self.max_pool(self.conv1(inputs))  # first cnn block
        x = self.max_pool(self.conv2(x))  # second cnn block
        x = self.flatten(x)  # flatten for dense nets
        x = self.dense3(self.dense2(self.dense1(x)))  # dense layers
        return x


# instantiate and compile the model
model = LeNet5(num_classes)
model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.predict(x_test[:10])

model.summary()

callbacks = [
    # callback to interrupt the training if the validation loss (val_loss) stops improving for over 3 epochs:
    tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss"),
    # callback to log the graph, losses and metrics into TensorBoard (saving log files in ./logs dir):
    tf.keras.callbacks.TensorBoard(
        log_dir="./logs", histogram_freq=1, write_graph=True
    ),
]

# now pass everything to the model and train it
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=80,
    validation_data=(x_train, y_train),
    verbose=1,
    callbacks=callbacks,
)

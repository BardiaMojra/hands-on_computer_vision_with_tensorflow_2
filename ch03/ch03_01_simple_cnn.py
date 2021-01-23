# ch03.01 - simple cnn, convolution on images
#%matplotlib inline
# !pip install scikit-image   # Uncomment to install this module
# !pip install matplotlib     # Uncomment to install this module
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io  # for reading images simply

image = io.imread("./in/bird_pic_by_benjamin_planche.png")

print("Image shape: {}".format(image.shape))
plt.figure()
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

image = tf.convert_to_tensor(image, tf.float32, name="input_image")

image = tf.expand_dims(
    image, axis=0
)  # we expand our tensor, adding a dimension at position 0

image = tf.expand_dims(
    image, axis=-1
)  # we expand our tensor, adding a dimension at position 0

print("Tensor shape: {}".format(image.shape))

# gaussian blur filter (kernel)
kernel = tf.constant(
    [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
    tf.float32,
    name="gaussian_kernel",
)

# need to be in shape (k, k, D, N)
kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

# run kernel on the image to "filter it"
blurred_img = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")

blurred_res = blurred_img.numpy()
# we "unbatch" our result by selecting the first (and only) image; we also remove the depth dimension:
blurred_res = blurred_res[0, ..., 0]

# show image
plt.figure()
plt.imshow(blurred_res, cmap=plt.cm.gray)
plt.show()


# contour detection kernel
kernel = tf.constant(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32, name="edge_kernel"
)
kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

edge_image = tf.nn.conv2d(image, kernel, strides=[1, 2, 2, 1], padding="SAME")
edge_res = edge_image.numpy()[0, ..., 0]  # unbatch and remove depth dimension

# show image
plt.figure()
plt.imshow(edge_res, cmap=plt.cm.gray)
plt.show()

# note edge result image has a white border cause by zero padding, it will
# disappear if we don't pad the image
edge_image = tf.nn.conv2d(image, kernel, strides=[1, 2, 2, 1], padding="VALID")
edge_res = edge_image.numpy()[0, ..., 0]
plt.figure()
plt.imshow(edge_res, cmap=plt.cm.gray)
plt.show()


# pooling
# for max-pooling and average-pooling, the values in each window is aggregated
# to a single value, max or average, respectively.

# average pooling
avg_pooled_img = tf.nn.avg_pool(
    image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
)
avg_res = avg_pooled_img.numpy()[0, ..., 0]
plt.figure()
plt.imshow(avg_res, cmap=plt.cm.gray)
plt.show()

# max pooling
max_pooled_img = tf.nn.max_pool(
    image, ksize=[1, 10, 10, 1], strides=[1, 2, 2, 1], padding="SAME"
)
max_res = max_pooled_img.numpy()[0, ..., 0]
plt.figure()
plt.imshow(max_res, cmap=plt.cm.gray)
plt.show()

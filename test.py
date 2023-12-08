# %%
import keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

# %%
# Image Config
HEIGHT = 320
WIDTH = 320
NUM_CLASSES = 1

# Augmentation Config
ROTATION_FACTOR = (-0.2, 0.2)

# Training Config
BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 3e-3
AUTOTUNE = tf.data.AUTOTUNE


# %%
def load_image(filepath: str):
    img = Image.open(filepath)
    img = img.resize((HEIGHT, WIDTH))
    img = np.array(img)
    img = np.cast['float32'](img)
    img = img / 255
    return img


def load_mask(filepath: str):
    img = Image.open(filepath)
    img = img.resize((HEIGHT, WIDTH))
    img = np.array(img)
    img = np.cast['float32'](img)
    img = img / 255
    return img


# %%
filelist_trainx = (glob.glob('data/inputx/*.jpg'))
images = np.array([load_image(fname) for fname in filelist_trainx])
filelist_trainy = (glob.glob('data/inputy/*.png'))
masks = np.array([load_mask(fname) for fname in filelist_trainy])


def random_rotation(x_image, y_image):
    angle = np.random.randint(-40, 40)
    rows, cols, _ = x_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    x_image = cv2.warpAffine(x_image, M, (cols, rows))
    y_image = cv2.warpAffine(y_image.astype('float32'), M, (cols, rows))
    return x_image, y_image.astype('int')


def horizontal_flip(x_image, y_image):
    x_image = cv2.flip(x_image, 1)
    y_image = cv2.flip(y_image.astype('float32'), 1)
    return x_image, y_image.astype('int')


def augment_data(images, masks):
    x_rotated = []
    y_rotated = []
    x_flipped = []
    y_flipped = []

    for i in range(len(images)):
        x_rot, y_rot = random_rotation(images[i], masks[i])
        x_flip, y_flip = horizontal_flip(images[i], masks[i])

        x_rotated.append(x_rot)
        y_rotated.append(y_rot)
        x_flipped.append(x_flip)
        y_flipped.append(y_flip)

    x_rotated = np.array(x_rotated)
    y_rotated = np.array(y_rotated)
    x_flipped = np.array(x_flipped)
    y_flipped = np.array(y_flipped)

    return np.concatenate((images, x_rotated, x_flipped)), np.concatenate((masks, y_rotated, y_flipped))


# SPlit the data into training and testing sets
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
# Augment the data
train_images, train_masks = augment_data(train_images, train_masks)
# %%
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

# Get the number of elements in the dataset
num_elements = len(list(dataset.as_numpy_iterator()))

# Define the size of the training set
train_size = int(0.8 * num_elements)

# Define the size of the testing set
test_size = num_elements - train_size

# Split the train into train and validation and shuffle
dataset = dataset.shuffle(buffer_size=num_elements)
train_dataset = dataset.take(test_size).batch(BATCH_SIZE)
validation_dataset = dataset.skip(test_size).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# %%
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(
        x
    )
    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# %%
import keras.backend as K


def jaccard_distance(y_true, y_pred, smooth=100):
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)


def dice_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    dice = (2 * intersection + smooth) / (sum_ + smooth)
    return dice


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())



# %%
# Build model
model = get_model(img_size=(HEIGHT, WIDTH), num_classes=NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss=[jaccard_distance]
    ,metrics=[dice_coef, sensitivity, specificity, accuracy]
)

checkpoint_filepath = 'model2/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model, doing validation at the end of each epoch.
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback]
)
for key in history.history.keys():
    print(key)


# Test model at the end on test data
model.evaluate(test_dataset)

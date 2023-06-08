import sys

from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard

IMG_SIZE = 224


# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    output = Dense(1, activation="sigmoid")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    # save plot to file
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory(f"formatted_data_{IMG_SIZE}/train/",
                                           class_mode="binary", batch_size=8, target_size=(IMG_SIZE, IMG_SIZE))
    test_it = datagen.flow_from_directory(f"formatted_data_{IMG_SIZE}/test/",
                                          class_mode="binary", batch_size=8, target_size=(IMG_SIZE, IMG_SIZE))

    print(len(train_it), len(test_it))

    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=2, verbose=1,
                        callbacks=[tensorboard])
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print("> %.3f" % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)
    model.save(f"models/{NAME}.h5")


if __name__ == "__main__":
    NAME = f"catsndogs-finalmodel-{IMG_SIZE}"
    run_test_harness()

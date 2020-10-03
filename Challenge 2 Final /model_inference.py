import sys
import os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras import optimizers

OUTPUT_PATH = "data/labelled.csv"


def main(input_path):
    # read data
    all_files = []
    for dirpath, dirnames, files in os.walk(input_path):
        if files:
            all_files.extend(files)

    df = pd.DataFrame(all_files)

    # IMPLEMENT DATA TRANSFORMATION HERE (IF REQUIRED)
    train_data_dir = 'train'
    validation_data_dir = 'val'
    img_width, img_height = 100, 100
    batch_size_train = 64
    batch_size_test = 32

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_train,
        class_mode='categorical',
        color_mode='grayscale')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_test,
        class_mode='categorical',
        color_mode='grayscale')
    # IMPLEMENT MODEL LOADING HERE
    def create_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        return model
    model = create_model()
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # IMPLEMENT MODEL INFERENCE HERE
    predictions = model.predict(files)
    prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')
    # ------ DUMMY TRANSFORMATION -------

    possible_predictions = ["anger", "annoyance", "doubt", "fatigue", "happiness", "sadness"]
    np.random.seed = 1
    dummy_predictions = np.random.choice(a=possible_predictions, size=len(df))
    dummy_predictions_df = pd.DataFrame(dummy_predictions)

    # --- END OF DUMMY TRANSFORMATION ---

    # SAVE PREDICTIONS TO CSV
    dummy_predictions_df = pd.concat([df, dummy_predictions_df], axis=1, sort=False)
    dummy_predictions_df.to_csv(OUTPUT_PATH, index=False, header=False)


if __name__ == "__main__":
    input_path = sys.argv[1]

    main(input_path)
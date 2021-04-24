

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)


midi_path = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_19r_step30/midi_slices'
audio_path = '/home/zackstrater/audio_midi_repository/denoise_no_sus_128mels_50l_19r_step30/audio_windows'
midi_files_bin = []
audio_files_bin = []
for filename in sorted(os.listdir(midi_path)):
    midi_files_bin.append(filename)

for filename in sorted(os.listdir(audio_path)):
    audio_files_bin.append(filename)

# for midi, audio in zip(midi_files_bin, audio_files_bin):
#     print(midi[0:-14])
#     print(audio[0:-15])
#     if midi[0:-14] != audio[0:-15]:
#         print('no')

y_images = []
for filename in midi_files_bin:
    array = np.load(os.path.join(midi_path, filename))
    y_images.append(array)
y_train = np.array(y_images)
# print(y_train.shape)
# print(y_train[0])
# print(audio_files_bin)
df = pd.DataFrame()
df['filenames'] = audio_files_bin
note_labels = np.arange(21, 109)
df[note_labels] = y_train
print(df)
# midi_img = Image.open(f'{midi_win_path}/{row.iloc[1]}')
#     # midi_arr = asarray(midi_img)/255

df_train, df_test = train_test_split(df, test_size=0.25)



gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(df_train, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                              seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,70))
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
valid_gen = valid_datagen.flow_from_dataframe(df_test, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                            seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,70))






def create_transfer_model(input_size, n_categories, weights='imagenet'):
    # note that the "top" is not included in the weights below
    base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)

    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    predictions = Dense(n_categories, activation='sigmoid')(model)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


transfer_model = create_transfer_model(input_size=(128, 100, 3), n_categories=88)


def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True


change_trainable_layers(transfer_model, 132)



transfer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
mdl_check_trans = ModelCheckpoint(filepath='../models/best_trans_model.hdf5',
                            save_best_only=True)
transfer_model.fit_generator(generator=train_small,
                    validation_data=val_small,
                    epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks=[mdl_check_trans])
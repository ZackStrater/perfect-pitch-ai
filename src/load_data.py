
import os
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


midi_path = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_27r_step20/midi_slices'
audio_path = '/home/zackstrater/audio_midi_repository/sus_128mels_100l_27r_step20/audio_windows'
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
df.to_csv('../data/sus_128mels_100l_27r_step20.csv', index=False)
print(df)

df_train, df_test = train_test_split(df, test_size=0.25)



gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_dataframe(df_train, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                              seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,128))
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_datagen.flow_from_dataframe(df_test, audio_path, x_col='filenames', y_col=note_labels, batch_size=32,
                                            seed=42, shuffle=True, class_mode='raw', color_mode='grayscale', target_size=(128,128))


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(128, 128, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(88, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])

model.fit(
        train_gen,
        steps_per_epoch=6168,
        epochs=10,
        validation_data=valid_gen,
        validation_steps=2056,
        verbose=1)
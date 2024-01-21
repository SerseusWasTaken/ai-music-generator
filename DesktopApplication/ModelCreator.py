import collections
import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

import keras_tuner as kt
import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from helperFunctions import midi_to_notes, notes_to_midi, mse_with_positive_pressure

class ModelCreator():

    def __init__(self) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self._SAMPLING_RATE = 16000
        self.key_order = ['pitch', 'step', 'duration']
        self.seq_length = 25
        self.vocab_size = 128
        self.batch_size = 64


    def __initDataset(self):
        self.data_dir = pathlib.Path('data/maestro-v3.0.0')
        if not self.data_dir.exists():
            tf.keras.utils.get_file(
                'maestro-v3.0.0-midi.zip',
                origin='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
                extract=True,
                cache_dir='.', cache_subdir='data',
            )
        self.filenames = glob.glob(str(self.data_dir/'**/*.mid*'))

    def create_sequences(
        self,
        dataset: tf.data.Dataset, 
        seq_length: int,
        vocab_size = 128,
    ) -> tf.data.Dataset:
        """Returns TF Dataset of sequence and label examples."""
        seq_length = seq_length+1

        # Take 1 extra for the labels
        windows = dataset.window(seq_length, shift=1, stride=1,
                                    drop_remainder=True)

        # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
        flatten = lambda x: x.batch(seq_length, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        # Normalize note pitch
        def scale_pitch(x):
            x = x/[vocab_size,1.0,1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key:labels_dense[i] for i,key in enumerate(self.key_order)}

            return scale_pitch(inputs), labels

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    def test_build_model(self):
        learning_rate = 0.005

        inputs = tf.keras.Input(input_shape, batch_size=self.batch_size)
        x = tf.keras.layers.LSTM(128, activation="sigmoid", stateful=True)(inputs)

        outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
        }

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            loss=loss,
            loss_weights={
                'pitch': 0.5,
                'step': 1.0,
                'duration':1.0,
            },
            optimizer=optimizer,
        )
        return model

    def createModel(self):
        self.__initDataset()
        num_files = 5
        all_notes = []
        for f in self.filenames[:num_files]:
            notes = midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)
        n_notes = len(all_notes)
        train_notes = np.stack([all_notes[key] for key in self.key_order], axis=1)
        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
        seq_ds = self.create_sequences(notes_ds, self.seq_length, self.vocab_size)
        buffer_size = n_notes - self.seq_length  

        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .batch(self.batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        #Global damit build_model darauf zugreifen kann
        global input_shape 
        input_shape = (self.seq_length, 3)
        global loss
        loss = {
            'pitch': SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,    
            'duration': mse_with_positive_pressure
        }
    

        model = self.test_build_model()

        epochs = 20

        history = model.fit(
            train_ds,
            epochs=epochs
        )
        model.save('models/tuner_best_mode.keras')

        




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

from helperFunctions import midi_to_notes, notes_to_midi

class ModelCreator():

    def __init__(self) -> None:
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self._SAMPLING_RATE = 16000
        self.key_order = ['pitch', 'step', 'duration']
        self.seq_length = 25
        self.vocab_size = 128


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
    

    def build_model(self, hp):
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(hp.Int('units', min_value=64, max_value=256, step=32))(inputs)  # Hier variieren wir die Anzahl der LSTM-Einheiten

        pitch_output = tf.keras.layers.Dense(128, name='pitch')(x)
        step_output = tf.keras.layers.Dense(1, name='step')(x)
        duration_output = tf.keras.layers.Dense(1, name='duration')(x)

        model = tf.keras.Model(inputs, [pitch_output, step_output, duration_output])

        # Kompilieren des Modells mit der optimierten Lernrate
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])  # Hier variieren wir die Lernrate
            )
            # ,metrics=['mean_squared_error', 'mean_absolute_error']
        )
        return model
    
    def test_build_model(self, hp):
        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(hp.Int('units', min_value=1028, max_value=2048, step=512))(inputs)  # Hier variieren wir die Anzahl der LSTM-Einheiten

        pitch_output = tf.keras.layers.Dense(128, name='pitch')(x)
        step_output = tf.keras.layers.Dense(1, name='step')(x)
        duration_output = tf.keras.layers.Dense(1, name='duration')(x)

        model = tf.keras.Model(inputs, [pitch_output, step_output, duration_output])

        # Kompilieren des Modells mit der optimierten Lernrate
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])  # Hier variieren wir die Lernrate
            )
            # ,metrics=['mean_squared_error', 'mean_absolute_error']
        )
        return model

    
    def print_statistics(self, loss_and_metrics):
        # Loss auf den Validierungsdaten ausgeben, sollte nahe 0 sein
        print(f'Verlust (Loss) auf den Validierungsdaten: {loss_and_metrics[0]}')

        # Weitere Metriken ausgeben (z. B. Mean Squared Error für Regressionsprobleme)
        print(f'Mean Squared Error (MSE) auf den Validierungsdaten: {loss_and_metrics[1]}')
        print(f'Mean Absolute Error (MAE) auf den Validierungsdaten: {loss_and_metrics[2]}')

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
        batch_size = 64
        buffer_size = n_notes - self.seq_length  
        # Aufteilung der Daten in Trainings- und Validierungsdaten | 80/20
        split = int(0.8 * buffer_size)

        # skip und take 80%
        train_ds = (seq_ds
                    .shuffle(buffer_size)
                    .take(split)  
                    .batch(batch_size, drop_remainder=True)
                    .cache()
                    .prefetch(tf.data.experimental.AUTOTUNE))

        validation_ds = (seq_ds
                        .shuffle(buffer_size)
                        .skip(split)
                        .batch(batch_size, drop_remainder=True)
                        .cache()
                        .prefetch(tf.data.experimental.AUTOTUNE))
        #Global damit buil_model darauf zugreifen kann
        global input_shape 
        input_shape = (self.seq_length, 3)
        global loss
        loss = {
            'pitch': SparseCategoricalCrossentropy(from_logits=True),
            'step': MeanSquaredError(),    
            'duration': MeanSquaredError()
        }
        learning_rate = 0.005
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        tuner = kt.Hyperband(
            self.test_build_model,
            objective='loss',
            max_epochs=20,
            hyperband_iterations=2
        )

        # Definition des Hyperparameter-Raums und Suche nach den besten Hyperparametern
        tuner.search(train_ds, validation_data = validation_ds)

        # Erhalten der besten Hyperparametern
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Erstellen des finalen Modells mit den besten Hyperparametern
        best_model = tuner.hypermodel.build(best_hps)

        best_model.save('models/tuner_best_mode.keras')
        best_model.summary()
        self.print_statistics(best_model.evaluate(validation_ds))




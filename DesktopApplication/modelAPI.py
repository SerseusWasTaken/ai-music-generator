import numpy as np
import tensorflow as tf
import pretty_midi
import collections
import pandas as pd
import os.path
from helperFunctions import midi_to_notes, notes_to_midi

class MusicRNN:
    def __init__(self, model, temperature, num_predictions) -> None:
        self.model = model
        self.temperature = temperature
        self.num_predictions = num_predictions
        self.key_order = ['pitch', 'step', 'duration']
        self.seq_length = 25
        self.vocab_size = 128
    
    def __predict_next_note(
        self,
        notes: np.ndarray, 
        model: tf.keras.Model, 
        temperature: float = 1.0) -> tuple[int, float, float]:
        """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions[0]
        step = predictions[1]
        duration = predictions[2]

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)


    def createMusicFile(self, sample_file):
        raw_notes = midi_to_notes(sample_file)
        pm = pretty_midi.PrettyMIDI(sample_file)
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        sample_notes = np.stack([raw_notes[key] for key in self.key_order], axis=1)
        input_notes = (sample_notes[:self.seq_length] / np.array([self.vocab_size, 1, 1]))

        generated_notes = []
        prev_start = 0
        for _ in range(self.num_predictions):
            pitch, step, duration = self.__predict_next_note(input_notes, self.model, self.temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(
            generated_notes, columns=(*self.key_order, 'start', 'end'))
        
        out_file = 'output.mid'
        out_pm = notes_to_midi(
            generated_notes, out_file=out_file, instrument_name=instrument_name)
        return out_pm

from modelAPI import MusicRNN
from createModel import ModelCreator
import tensorflow as tf
import pathlib
import glob
import keras
from helperFunctions import mse_with_positive_pressure

def main():
    creator = ModelCreator()
    creator.createModel()

    """
    model = tf.keras.saving.load_model('models/tuner_best_mode.keras')
    modelAPI = MusicRNN(model, 2.0, 128)

    data_dir = pathlib.Path('data/maestro-v3.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v3.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )
    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    sample_file = filenames[1]

    modelAPI.createMusicFile(sample_file)"""


    
    

if __name__ == "__main__":
    main()
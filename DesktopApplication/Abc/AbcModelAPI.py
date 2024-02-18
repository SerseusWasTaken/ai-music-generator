import tensorflow as tf 
import numpy as np
from tqdm import tqdm

from mitdeeplearning import lab1
from Abc.AbcModel import build_model


class AbcModelAPI():
    def __init__(self, checkpoint_dir) -> None:
        songs = lab1.load_training_data()
        songs_joined = "\n\n".join(songs) 
        vocab = sorted(set(songs_joined))
        self.char2idx = {u:i for i, u in enumerate(vocab)}
        self.idx2char = np.array(vocab)
        # these must be adjusted if adjusted in AbcModel.py
        vocab_size = len(vocab) 
        embedding_dim = 256 
        rnn_units = 1024  
        self.model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        

    def __generate_text(self, model, start_string, generation_length=1000):
        # Evaluation step (generating ABC text using the learned RNN model)

        input_eval = [self.char2idx[num] for num in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        model.reset_states()
        tqdm._instances.clear()

        for i in tqdm(range(generation_length)):
            predictions = model(input_eval)
            
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            # Pass the prediction along with the previous hidden state
            #   as the next inputs to the model
            input_eval = tf.expand_dims([predicted_id], 0)
            
            # Hint: consider what format the prediction is in vs. the output
            text_generated.append(self.idx2char[predicted_id])
            
        return (start_string + ''.join(text_generated))
    
    def generate_song(self):
        num_of_songs = 0
        generated_songs = 0
        # repeat until at least one song is generated
        while(num_of_songs == 0):
            generated_text = self.__generate_text(
                self.model, start_string="X", generation_length=1000
                )
            generated_songs = lab1.extract_song_snippet(generated_text)
            num_of_songs = len(generated_songs)
            print(num_of_songs)

        lab1.play_song(generated_songs[0])

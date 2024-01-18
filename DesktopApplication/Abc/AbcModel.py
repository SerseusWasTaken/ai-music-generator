import tensorflow as tf 
import numpy as np
import os
from tqdm import tqdm

from mitdeeplearning import lab1

#From https://github.com/mertbozkir/Music_Generation_RNNs/blob/main/Part2_Music_Generation.ipynb

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
  output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors 
        #   of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units. 
        tf.keras.layers.LSTM(
            rnn_units, 
            return_sequences=True, 
            recurrent_initializer='glorot_uniform',
            recurrent_activation='sigmoid',
            stateful=True,
        ),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output into the vocabulary size. 
        tf.keras.layers.Dense(units = vocab_size)
    ])
    return model


def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss

@tf.function
def train_step(x, y, model, optimizer): 
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)

        loss = compute_loss(y, y_hat)

    # Now, compute the gradients 
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def createABCModel():
    songs = lab1.load_training_data()
    # Join our list of song strings into a single string containing all songs
    songs_joined = "\n\n".join(songs) 

    # Find all unique characters in the joined string
    vocab = sorted(set(songs_joined))
    # Create a mapping from character to unique index.
    # For example, to get the index of the character "d", 
    #   we can evaluate `char2idx["d"]`.  
    char2idx = {u:i for i, u in enumerate(vocab)}

    # Create a mapping from indices to characters. This is
    #   the inverse of char2idx and allows us to convert back
    #   from unique index to the character in our vocabulary.
    idx2char = np.array(vocab)

    def vectorize_string(string):
        vectorized_songs = np.array([char2idx[song] for song in string ])
        return vectorized_songs
    
    vectorized_songs = vectorize_string(songs_joined)
    x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

    ### Hyperparameter setting and optimization ###

    # Optimization parameters:
    num_training_iterations = 2000  # Increase this to train longer
    batch_size = 4  # between 1 and 64
    seq_length = 100  #  between 50 and 500
    learning_rate = 5e-3  #  between 1e-5 and 1e-1

    # Model parameters: 
    vocab_size = len(vocab) # 83 at the moment
    embedding_dim = 256 
    rnn_units = 1024  #  between 1 and 2048

    # Checkpoint location: 
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    ### Training ### 
    history = []
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for iter in tqdm(range(num_training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(x_batch, y_batch, model, optimizer)

        # Update the progress bar
        history.append(loss.numpy().mean())

        # Update the model with the changed weights!
        if iter % 100 == 0:     
            model.save_weights(checkpoint_prefix)
        
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)


def main():
    createABCModel()

if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import ttk
#Music generation related
from helperFunctions import mse_with_positive_pressure
import keras
import tensorflow as tf
from modelAPI import MusicRNN
import pathlib
import glob

def generateMusicFile():
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

    modelAPI.createMusicFile(sample_file)

# creates Window
root = tk.Tk()

root.title("Little Conductor")
# start size window + where opens 
root.geometry("1200x800+100+100") 
root.minsize(400, 300)  # width, height
root.configure(background="#4f6367")

frame = tk.Frame(root)
frame.pack()

# Create Label in our window
# label with a specific font
header = ttk.Label(
    root,
    text='Little Conductor',
    font=("Amatic SC", 45),
    background='#4f6367',
    padding=0,
    foreground='white')

header.pack()

devNames = ttk.Label(
    root,
    text='Developer: Bach, Sailer, Schlecht ',
    font=("Comfortaa", 10),
    background='#4f6367',
    foreground='white')

devNames.pack()

button = ttk.Button(
    root,
    text="Create a masterpiece",
    command=lambda: generateMusicFile()
)
button.pack()


# use a GIF image you have in the working directory
# or give full path
photo = tk.PhotoImage(file="./assets/conductorWithBackground.png")

tk.Label(root, image=photo,border=0).pack()



# keeps windwo visible
root.mainloop()

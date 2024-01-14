import tkinter as tk
from tkinter import HORIZONTAL, ttk
from pygame import mixer
#Music generation related
from helperFunctions import mse_with_positive_pressure
import keras
import tensorflow as tf
from modelAPI import MusicRNN
import pathlib
import glob
from customtkinter import *

def generateMusicFile():
    #todo: prüfe ob path bei allen richtig oder ob model in assets holen
    model = tf.keras.saving.load_model('../models/tuner_best_mode.keras')
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

def generateAndPlay():
    generateMusicFile()
    openPlayer()

def playMusic(path):
    mixer.music.load(path)
    mixer.music.play()
    btn_play.grid_forget()
    btn_pause.grid(row=0, column=0, padx=10, pady=10)

def pauseMusic():
    mixer.music.pause()
    btn_pause.grid_forget()
    btn_play.grid(row=0, column=0, padx=10, pady=10)

def unmute():
    mixer.music.set_volume(100.0)
    btn_unmute.grid_forget()
    btn_mute.grid(row=0, column=1, padx=10, pady=10)

def mute():
    mixer.music.set_volume(0.0)
    btn_mute.grid_forget()
    btn_unmute.grid(row=0, column=1, padx=10, pady=10)

def buttonFrame():
    inner.pack()

def openPlayer():
    buttonFrame()
    pauseMusic()
    btn_h.place(x=10, y=10)
    frame1.pack_forget()
    frame2.pack(fill=tk.BOTH)

def showHome():
    btn_h.place_forget()
    pauseMusic()
    frame2.pack_forget()
    frame1.pack()

def homeFrame():
    s = ttk.Style()
    s.configure('my.TButton', font=('Helvetica', 22))
    ttk.Button(
        frame1,
        style='my.TButton',
        text="Create a masterpiece",
        command=lambda: generateAndPlay()
    ).pack()
    showHome()

# creates Window
root = tk.Tk()
root.title("Little Conductor")

# start size window + where opens 
root.geometry("1200x800+100+100") 
root.minsize(400, 300)  # width, height
root.configure(background="#4f6367")

#create a Frame for Home and for the Audio Player
frame1 = tk.Frame(root, background="#4f6367")
frame2 = tk.Frame(root, background="#4f6367")

#insert some conent to the root
btn_home = tk.PhotoImage(file="./assets/home.png")
btn_h = ttk.Button(root, image=btn_home, command=lambda: showHome(), compound=tk.CENTER)

ttk.Label(
    root,
    text='Developer: Bach, Sailer, Schlecht ',
    font=("Comfortaa", 10),
    background='#4f6367',
    foreground='white').pack()

conducter_img = tk.PhotoImage(file="./assets/conductorWithBackground.png")
tk.Label(root, image=conducter_img, border=0).pack()

ttk.Label(
    root,
    text='Little Conductor',
    font=("Amatic SC", 45),
    background='#4f6367',
    padding=0,
    foreground='white').pack()

#initalizes the mixer
mixer.init()

#create Pause and Play Buttons
pause_img = tk.PhotoImage(file="./assets/pause.png")
play_img = tk.PhotoImage(file="./assets/play.png")
mute_img = tk.PhotoImage(file="./assets/mute.png")
unmute_img = tk.PhotoImage(file="./assets/unmute.png")

#put them into a smaller Frame
inner = tk.Frame(frame2, width=140, height=70, relief='solid', borderwidth=0)
btn_pause = ttk.Button(inner, image=pause_img, command=pauseMusic, compound=tk.CENTER)
btn_play = ttk.Button(inner, image=play_img, command=lambda: playMusic("output.mid"), compound=tk.CENTER)
btn_mute = ttk.Button(inner, image=unmute_img, command=lambda: mute(), compound=tk.CENTER)
btn_unmute = ttk.Button(inner, image=mute_img, command=lambda: unmute(), compound=tk.CENTER)
btn_play.grid(row=0, column=0, padx=10, pady=10)
btn_mute.grid(row=0, column=1, padx=10, pady=10)

homeFrame()
# keeps windwo visible
root.mainloop()
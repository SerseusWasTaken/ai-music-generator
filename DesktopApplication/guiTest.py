from customtkinter import *
from PIL import Image
import tkinter as tk
from tkinter import HORIZONTAL, ttk
from pygame import mixer
#Music generation related
from helperFunctions import mse_with_positive_pressure
import tensorflow as tf
from modelAPI import MusicRNN
import pathlib
import glob
from customtkinter import *
import threading
from tkinter import PhotoImage

from Abc.AbcModelAPI import AbcModelAPI

#Model management
use_abc_model = False
abcModel = AbcModelAPI('./abc/training_checkpoints')

#Variablen User
temperature = 1

mixer.init()

#Window
app = CTk(fg_color="#4f6367")
app.geometry("1200x800+100+100")
app.minsize(800, 800)  # width, height
app.title("Little Conductor")
#app.iconbitmap(default="assets\icon.ico")

set_appearance_mode("light")

# Frames 
homeFrame = CTkFrame(master=app, width=1200, height=800, fg_color="#4f6367")
homeFrame.pack(expand=False)
loadingFrame = CTkFrame(master=app, width=1200, height=800, fg_color="#4f6367")
playerFrame = CTkFrame(master=app, width=1200, height=800, fg_color="#4f6367")

#Methoden
def generateUsingAbcModel():
    global abcModel
    abcModel.generate_song()
    switchToPlayer()

def generateUsingMIDIModel():
    global temperature
    #todo: prüfe ob path bei allen richtig oder ob model in assets holen
    model = tf.keras.saving.load_model('../models/tuner_best_mode.keras')
    modelAPI = MusicRNN(model, temperature, 128)

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

    switchToPlayer()

# todo: 
def slider_event(value):
    global temperature
    temperature = value

def generateButton():
    global use_abc_model
    homeFrame.pack_forget()
    loadingFrame.pack()

    if use_abc_model:
        t = threading.Thread(target=generateUsingAbcModel)
        t.start()
    else:
        t = threading.Thread(target=generateUsingMIDIModel)
        t.start()

def playMusic():
    path = ''
    if use_abc_model: 
        path = 'tmp.wav'
    else:
        path = 'output.mid'

    mixer.music.load(path)
    mixer.music.play()

    btnPlay.grid_forget()
    btnPause.grid(row=0, column=0, padx=10, pady=10)

def pauseMusic():
    mixer.music.pause()

    btnPause.grid_forget()
    btnPlay.grid(row=0, column=0, padx=10, pady=10)

def switchToPlayer():
    progressbar.stop()
    loadingFrame.pack_forget()
    playerFrame.pack(fill=tk.BOTH)
    pauseMusic()

def unmute():
    mixer.music.set_volume(100.0)
    btnUnmute.grid_forget()
    btnMute.grid(row=0, column=1, padx=10, pady=10)

def mute():
    mixer.music.set_volume(0.0)
    btnMute.grid_forget()
    btnUnmute.grid(row=0, column=1, padx=10, pady=10)

def home():
    playerFrame.pack_forget()
    inner.grid_forget()
    homeFrame.pack()

#todo: switch Model
def switchModel():
    global use_abc_model
    use_abc_model = not use_abc_model

#Define  Interface Home
dev = CTkLabel(homeFrame, text="Developer: Bach, Sailer, Schlecht", text_color="white", font=("Comfortaa", 13), bg_color="#4f6367", height=10)

sliderText = CTkLabel(homeFrame, text="Stelle ein wie 'expermentierfreudig' das Modell ist: ", text_color="white", font=("Comfortaa", 13), bg_color="#4f6367")

btn = CTkButton(master=homeFrame, text="Generate Masterpiece", corner_radius=20, font=("Amatic SC", 45),
                fg_color="#D9D9D9", hover_color="#EEC6C6", text_color="#FE5F55", width=600, height=100, command=generateButton)

sliderTemperature = CTkSlider(homeFrame, from_=0.1, to=5, command=slider_event, button_color="#FE5F55", progress_color="#FE5F55", height=20, width=400)
sliderTemperature.set(1)

conductorImg = CTkImage(light_image=Image.open("./assets/conductorWithBackground.png"), size=(270, 270))
conductorLabelImg = CTkLabel(master=homeFrame, image=conductorImg, text="")

title = CTkLabel(homeFrame, text='Little Conductor', font=("Amatic SC", 200),bg_color='#4f6367', text_color="white")

currentModel = StringVar(value="off")
switch = CTkSwitch(homeFrame, command=switchModel, variable=currentModel, onvalue="on", offvalue="off", text="Use ABC Model", progress_color="#FE5F55", text_color="white")

#Home order:
conductorLabelImg.place(relx= 0.5, rely= 0.9, anchor="center")
btn.place(relx=0.5, rely=0.55, anchor="center")
dev.place(relx=0.5, rely=0.08, anchor="center")
sliderText.place(relx=0.5, rely=0.65, anchor="center")
sliderTemperature.place(relx=0.5, rely=0.7, anchor="center")
title.place(relx= 0.5, rely= 0.3, anchor="center")
switch.place(relx=0.5, rely=0.63, anchor="center")

#Define Loading 
progressbar = CTkProgressBar(loadingFrame, orientation="horizontal", height=40, width=750, progress_color="#FE5F55", )
progressbar.configure(mode="indeterminate")
progressbar.start()

#Loading order:
progressbar.place(relx=0.5, rely=0.8, anchor="center")

#Define Play 
pause_img = PhotoImage(file="./assets/pause.png")
play_img = PhotoImage(file="./assets/play.png")
mute_img = PhotoImage(file="./assets/mute.png")
unmute_img = PhotoImage(file="./assets/unmute.png")
home_img = PhotoImage(file="./assets/home.png")

inner = CTkFrame(playerFrame, width=140, height=70)

#hover_color="#EEC6C6",
#fg_color="#4f6367",  text_color="#FE5F55",
btnPause = CTkButton(master = inner, image=pause_img,command=pauseMusic, bg_color="transparent", fg_color="transparent", hover_color="#87A9B0",  height=50, text="")
btnMute = CTkButton(master = inner, image=mute_img,  command=mute,  bg_color="transparent", fg_color="transparent", hover_color="#87A9B0", height=50, text="")
btnPlay = CTkButton(master = inner, image=play_img, command=lambda: playMusic(), bg_color="transparent", fg_color="transparent", hover_color="#87A9B0", height=50, text="")
btnUnmute = CTkButton(master = inner, image=unmute_img, command=unmute, bg_color="transparent", fg_color="transparent",hover_color="#87A9B0",  height=50, text="")

# Mute/Unmute und Play/Pause switchen wenn gedrückt
btnPause.grid(row=0, column=0, padx=10, pady=10)
btnMute.grid(row=0, column=1, padx=10, pady=10)

inner.grid(row=0, column=0, padx=440, pady=440) 

conductorImg = CTkImage(light_image=Image.open("./assets/conductorWithBackground.png"), size=(250, 250))
conductorLabelImg = CTkLabel(master=playerFrame, image=conductorImg, text="")
conductorLabelImg.place(relx= 0.5, rely= 0.3, anchor="center")

btnHome = CTkButton(playerFrame, image=home_img, command=home, text="", bg_color="transparent",  hover_color="#87A9B0", fg_color="transparent")
btnHome.place(x=10, y=10)

app.mainloop()
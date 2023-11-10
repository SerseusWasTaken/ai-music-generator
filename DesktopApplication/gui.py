import tkinter as tk
from tkinter import ttk


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
    text="Create a masterpiece"
)
button.pack()


# use a GIF image you have in the working directory
# or give full path
photo = tk.PhotoImage(file="./assets/conductorWithBackground.png")

tk.Label(root, image=photo,border=0).pack()



# keeps windwo visible
root.mainloop()

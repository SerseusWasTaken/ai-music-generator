# Repository for the "KI für Musik" class


## Setup

## 1. Install FluidSynth  
for example through 
```choco install fluidsynth```

## 2. Install PyFluidSynth

Download https://github.com/nwhitehead/pyfluidsynth/archive/master.zip  
Open the folder (.../pyfluidsynth-master/pyfluidsynth-master) in cmd and use follwing command  
`python setup.py install`   

Download the File libfluidsynth64.dll [here](https://github.com/fkortsagin/Heretic-Shadow-of-the-Serpent-Riders-Windows-10/blob/master/libfluidsynth64.dll)  
Rename the file to libfluidsynth.dll  
Place it into C:\Windows\System32  

Place the file `fluidsynth.py` from the zip folder (.../pyfluidsynth-master/pyfluidsynth-master) into the project

## 3. Run the project and install the other dependencies
tip: restart kernel after installing a dependency with pip install
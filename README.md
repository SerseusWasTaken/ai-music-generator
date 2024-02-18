# Repository for the "KI für Musik" class


## Setup and requirements

## 1. Install FluidSynth
For example for Windows with  
```
choco install fluidsynth
```  
Or MacOS  
```
brew install fluid-synth
```


## 2. Install PyFluidSynth (Windows)
Note: this may not be needed if the file in the repository does already work for you.   
Download https://github.com/nwhitehead/pyfluidsynth/archive/master.zip  
Open the folder (.../pyfluidsynth-master/pyfluidsynth-master) in cmd and use follwing command  
`python setup.py install`   

Download the File libfluidsynth64.dll [here](https://github.com/fkortsagin/Heretic-Shadow-of-the-Serpent-Riders-Windows-10/blob/master/libfluidsynth64.dll)  
Rename the file to libfluidsynth.dll  
Place it into C:\Windows\System32  


## Install other needed system packages
- [timidity](https://formulae.brew.sh/formula/timidity)
- [abcmidi](https://formulae.brew.sh/formula/abcmidi)
- [python-tk](https://formulae.brew.sh/formula/python-tk@3.11)

## Install python dependencies
Run `pip install -r requirements.txt`  

## Train the models
- For the ABC model, go into the [DesktopApplication/Abc](DesktopApplication/Abc) directory and execute `python ./AbcModel.py`
- For the Midi model, go into the [DesktopApplication/Midi](DesktopApplication/Midi) directory and execute `python ./ModelCreator.py`

## Run the GUI
Go into the [DesktopApplication](DesktopApplication) directory and run `python GUI.py`
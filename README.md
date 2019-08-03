# ScreenTools
Python package for image data processing at the Argonne Wakefield Accelerator


WORKFLOW

- CONVERT FILES INTO HDF5 USING METHODS IN file_processing.py
  - WILL ASK YOU TO SELECT SCREEN INNER EDGE TO DO PIXEL SCALING
  - IF FOLDER HAS MANY FILES OF SAME SCREEN THEN CHOOSE SAME_SCREEN = TRUE IN PROCESS_FOLDER
- PROCESS IMAGES USING METHODS IN image_processing.py
- ANALYZE IMAGES USING METHODS IN analyze.py
- PLOT FILES USING plotting.py


FILE STRUCTURE
--------------
- / (ROOT,GROUP)
  -ROOT ATTRIBUTES (METADATA)
  - /1 (SHOT NUMBER,GROUP)
    - SHOT ATTRIBUTES (CHARGE,MEASURED RMS ETC.)
    - \img (PROCESSED IMAGE,DATASET)
    - \raw (RAW IMAGE FOR RESETTING, DATASET)
  - /2
  - /3
  .
  .
  - /N
  
  
  
EXAMPLE SCRIPT (READ IN DATA, DATA OUTSIDE SCREEN, PLOT FIRST SCREEN IMAGE
--------------------------------------------------------------------------
def main()
    file_process.process_folder(folder)
    frame_number = 0
    image_processing.mask(filename,frame_number)
    plotting.plot_screen(fname,frame_number=frame_number)

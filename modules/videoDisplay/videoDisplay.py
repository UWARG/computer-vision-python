# libraries
#from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker

import cv2
from matplotlib.font_manager import is_opentype_cff_font
import numpy as np
import multiprocessing as mp
import logging
import time

is_open = None
# method that displays webcam
def displayCamera():
  cap = cv2.VideoCapture(0)
  global is_open
  is_open = False

  if (cap.isOpened() == False):
    print("Unable to read camera feed")
  # Default resolutions of the frame are obtained.The default resolutions are system dependent
  # We convert the resolutions from float to integer
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))

  # Define the codec and create VideoWriter object
  out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

  timeout = 5 #5 seconds
  timeout_start = time.time()

  while(True):
    ret, frame = cap.read()
    is_open = True

    if ret == True:
      out.write(frame)

      cv2.imshow('frame',frame)

      if time.time() > timeout_start + timeout:
        break

      if cv2.waitKey(1) & 0xFF == ord('q'): # close the window when 'q' is clicked
        break

    # Break the loop
    else:
      break

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1: # close the window when top right 'X' is clicked
        break

  is_open = False

  cap.release()
  out.release()

  # Closes all the frames
  cv2.destroyAllWindows()


def displayVideo(pause, exitRequest, frameIn): # this function needs to take in pipelineOut from decklinkSrcWorker and display to screen as window popup
    logger = logging.getLogger()
    logger.debug("videoDisplay: Started video display")

    

    logger.debug("videoDisplay: Stopped video display")

if __name__ == "__main__":
  displayCamera()

# libraries
import cv2
import logging

def videoDisplayWorker(pause, exitRequest, pipelineIn):
  logger = logging.getLogger()
  logger.debug("videoDisplay: Started video display")

  while True:

    if not exitRequest.empty():
      cv2.destroyAllWindows()
      break

    pause.acquire()
    pause.release()

    current_frame = pipelineIn.get()

    if current_frame is None:
      continue

    cv2.imshow('Frame', current_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      exitRequest = 1
      break

    # cap = cv2.VideoCapture()
        
    # if (cap.isOpened()== False): 
    #   print("Error opening video")
        
    # while(cap.isOpened()):
            
    #   # Capture frame-by-frame
    #   ret, frame = cap.read()
    #   if ret == True:
        
    #     # Display the resulting frame
    #     cv2.imshow('Frame', curr_frame)
        
    #     # press q to close
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #       break
          
    #   else: 
    #     break

  # here, add displaying the curr_frame to window

  logger.debug("videoDisplay: Stopped video display")
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

  logger.debug("videoDisplay: Stopped video display")
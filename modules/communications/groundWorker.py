from modules.communications.groundSend import GroundSendRecieve
import logging


def flt_conn_send_worker(pause, exitRequest, pipelineIn, pipelineOut):
    logger = logging.getLogger()
    logger.debug("ground_send_worker: Ground Send Worker started")

    ground_send_recieve = GroundSendRecieve()

    while True:

        if not exitRequest.empty():
            break

        pause.acquire()
        pause.release()

        send_request = pipelineIn.get(block=True)

        gijo_string = ground_send_recieve.encode(send_request)
        ground_send_recieve.send(gijo_string)

        while len(ground_send_recieve.decoded_q) != 0:
            pipelineOut.put(ground_send_recieve.decoded_q.pop())

    logger.debug("ground_send_worker: Ground Send Worker stopped")
from modules.communications.FltConnSend import FltConnSend
import logging


def flt_conn_send_worker(pause, exitRequest, pipelineIn):
    logger = logging.getLogger()
    logger.debug("flight_controller_send_worker: Flight Controller Send Worker started")

    fltConnSend = FltConnSend()

    FIJO_BUILDERS = {
        "QR_COORDINATES": fltConnSend.build_fijo_for_qr
    }

    while True:

        if not exitRequest.empty():
            break

        pause.acquire()
        pause.release()

        send_request = pipelineIn.get(block=True)

        fijo_string = FIJO_BUILDERS[send_request['type']](send_request)
        fltConnSend.send_fijo(fijo_string)

    logger.debug("flight_controller_send_worker: Flight Controller Send Worker stopped")

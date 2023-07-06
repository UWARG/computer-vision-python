"""
Gets data from ZeroPilot
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import zp_input


def zp_input_worker(port: str,
                    baudrate: int,
                    telemetry_output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                    request_output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                    controller: worker_controller.WorkerController):
    """
    Worker process.

    port is UART port.
    baudrate is UART baudrate.
    telemetry_output_queue is the telemetry queue.
    request_output_queue is the ZP request queue.
    controller is how the main process communicates to this worker process.
    """
    input_device = zp_input.ZpInput(port, baudrate)

    while not controller.is_exit_requested():
        controller.check_pause()

        result, value = input_device.run()
        if not result:
            continue

        # Get Pylance to stop complaining
        assert value is not None
        assert value.message is not None

        # Decide which worker to send to next depending on message type
        if value.message.header.type == 0:
            # Odometry
            telemetry_output_queue.queue.put(value)
        elif value.message.header.type == 1:
            # Request
            request_output_queue.queue.put(value)
        else:
            # TODO: Invalid type, log it?
            pass

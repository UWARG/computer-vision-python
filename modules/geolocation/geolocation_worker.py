"""
Convert bounding box data into ground data.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import camera_properties
from . import geolocation


# Extra parameters required for worker communication
# pylint: disable=too-many-arguments
def geolocation_worker(camera_intrinsics: camera_properties.CameraIntrinsics,
                       camera_drone_extrinsics: camera_properties.CameraDroneExtrinsics,
                       input_queue: queue_proxy_wrapper.QueueProxyWrapper,
                       output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                       controller: worker_controller.WorkerController):
    """
    Worker process.

    input_queue and output_queue are data queues.
    controller is how the main process communicates to this worker process.
    """
    # TODO: Logging?
    # TODO: Handle errors better

    result, locator = geolocation.Geolocation.create(
        camera_intrinsics,
        camera_drone_extrinsics,
    )
    if not result:
        return

    # Get Pylance to stop complaining
    assert locator is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            break

        result, value = locator.run(input_data)
        if not result:
            continue

        output_queue.queue.put(value)

# pylint: enable=too-many-arguments

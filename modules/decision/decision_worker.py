"""
Retrieves list of landing pads from cluster estimation and outputs decision to the flight controller.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import decision
from . import landing_pad_tracking
from . import search_pattern


def decision_worker(
    distance_squared_threshold: float,
    tolerance: float,
    camera_fov_forwards: float,
    camera_fov_sideways: float,
    search_height: float,
    search_overlap: float,
    small_adjustment: float,
    odometry_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    cluster_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    PARAMETERS
    ----------
        - camera_fov_forwards, camera_fov_sideways, search_height, search_overlap, distance_squared_threshold, 
          and small_adjustment are arguments for the constructors below.
        - cluster_input_queue and output_queue are the data queues.
        - controller is how the main process communicates to this worker.
    """

    landing_pads = landing_pad_tracking.LandingPadTracking(distance_squared_threshold)
    decision_maker = decision.Decision(tolerance)
    search = search_pattern.SearchPattern(
        camera_fov_forwards,
        camera_fov_sideways,
        search_height,
        search_overlap,
        distance_squared_threshold=distance_squared_threshold,
        small_adjustment=small_adjustment,
    )

    while not controller.is_exit_requested():
        controller.check_pause()
        
        curr_state = odometry_input_queue.queue.get_nowait()

        if curr_state is None:
            continue

        input_data = cluster_input_queue.queue.get()

        if input_data is None:
            continue

        is_found, best_landing_pads = landing_pads.run(input_data)

        # Runs decision only if there exists a landing pad
        if not is_found:
            result, value = search.continue_search(curr_state)
        else:
            result, value = decision_maker.run(curr_state, best_landing_pads)

        if not result:
            continue

        output_queue.queue.put(value)

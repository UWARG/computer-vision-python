"""
Retrieves list of landing pads from cluster estimation and outputs decision to the flight controller 

"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import decision
from . import landing_pad_tracking
from . import search_pattern
from .. flight_interface.flight_interface_worker import current_odometry, odometry_mutex


def decision_worker(
        camera_fov_forwards: float,
        camera_fov_sideways: float,
        search_height: float,
        search_overlap: float,
        distance_squared_threshold: float,
        small_adjustment: float,
        tolerance: float,
        cluster_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
        output_queue: queue_proxy_wrapper.QueueProxyWrapper,
        controller: worker_controller.WorkerController
    ) -> None: 
    """
    Worker process.

    PARAMETERS
    ----------
        - camera_fov_forwards and camera_fov_sideways are the measurements for the cameras field of view 
        - search_height and search overlap are the parameters for the search pattern
        - distance_squared_threshold, small adjustment, and tolerance are the initial settings
        - cluster_input_queue and output_queue are the data queues.
        - conteroller is how the main process communicates to this worker.
    """

    landing_pads = landing_pad_tracking.LandingPadTracking(distance_squared_threshold)
    decision_maker = decision.Decision(tolerance)
    search = search_pattern.SearchPattern(
        camera_fov_forwards,
        camera_fov_sideways,
        search_height,
        search_overlap,
        distance_squared_threshold=distance_squared_threshold,
        small_adjustment=small_adjustment
    )

    
    while not controller.is_exit_requested():
        controller.check_pause()

        #Accesses mutex to get latest odometry data
        with odometry_mutex:
            curr_state = current_odometry

        if curr_state is None:
            break 

        input_data = cluster_input_queue.queue.get()

        if input_data is None:
            break

        found, best_landing_pads = landing_pads.run(input_data)

        #Runs decision only if there exists a landing pad
        if not found:
            result, value = search.continue_search(curr_state)
        else:
            result, value = decision_maker.run(curr_state,best_landing_pads)

        if not result:
            break 

        output_queue.queue.put(value)



    

    
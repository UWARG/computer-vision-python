"""
Auto-landing worker.
"""
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import auto_landing

def auto_landing_worker(
    FOV_X: float,
    FOV_Y: float,
    im_h: float,
    im_w: float,
    x_center: float,
    y_center: float,
    height: float,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
    ) -> None:

    auto_lander = auto_landing.AutoLanding(FOV_X, FOV_Y, im_h, im_w)
    """
    result, lander = cluster_estimation.ClusterEstimation.create(
        min_activation_threshold,
        min_new_points_to_run,
        random_state,
    )
    currently don't have a create function
    """
    if not result:
        print("ERROR: Worker failed to create class object")
        return

    # Get Pylance to stop complaining
    assert auto_lander is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            continue

        result, value = auto_lander.run(input_data, False)
        if not result:
            continue

        output_queue.queue.put(value)
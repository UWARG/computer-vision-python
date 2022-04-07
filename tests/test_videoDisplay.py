import modules.videoDisplay.videoDisplay as vd
import logging

LOGGER = logging.getLogger()


def test_camera_display():
    vd.displayCamera()
    assert vd.is_open is False


def test_log_output(caplog):
    with caplog.at_level(logging.DEBUG):
        vd.displayVideo(None, None, None)
        assert 'Started video display' in caplog.text
        assert 'Stopped video display' in caplog.text

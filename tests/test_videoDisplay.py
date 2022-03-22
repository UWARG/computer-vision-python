import modules.videoDisplay.videoDisplay as vd
import unittest


class TestVideoDisplay(unittest.TestCase):
    def test_camera_display(self):
        vd.displayCamera()
        assert vd.is_open is False

    def test_log_output(self):
        with self.assertLogs(level='DEBUG') as cm:
            vd.displayVideo(None, None, None)
            self.assertEqual(cm.output, ['DEBUG:root:videoDisplay: Started video display',
                                         'DEBUG:root:videoDisplay: Stopped video display'])

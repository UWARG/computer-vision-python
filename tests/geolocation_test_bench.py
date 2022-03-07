import configparser
import modules.TestIMUInterface.getIMUData as getIMUData
import modules.decklinksrc.decklinksrc as decklinksrc
import cv2
import modules.geolocation.geolocation as geolocation
import numpy as np


def locate(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        ret, result = locator.run_locator(telemetry, np.array([[x, y]]))
        if (ret == True):
            print(f"Pixel {x},{y} maps to world {result[0]}")
        else:
            print("Geolocation error!")


if __name__ == "__main__":

    # Config file
    config = configparser.ConfigParser()
    config.read("geolocation_test_bench.ini")

    # GPS coordinates
    coordinates = dict(
        altitude = float(config["gpsCoordinates"]["altitude"]),
        longitude = float(config["gpsCoordinates"]["longitude"]),
        latitude = float(config["gpsCoordinates"]["latitude"])
    )

    # Euler angles of plane
    comPort = config["comPort"]["port"]
    getIMUWorker = getIMUData.getIMUInterface(comPort)

    xyz = getIMUWorker.getIMUData()
    print(xyz)
    planeAngles = getIMUWorker.ZYXFromIMU(xyz)

    planeAngles["yaw"] = float(config["eulerAnglesOfPlane"]["yaw"])

    # Euler angles of camera (with respect to plane)
    # Advanced stuff
    cameraAngles = dict(
        yaw = float(config["eulerAnglesOfCamera"]["yaw"]),
        pitch = float(config["eulerAnglesOfCamera"]["pitch"]),
        roll = float(config["eulerAnglesOfCamera"]["roll"])
    )

    telemetry = dict(
        gpsCoordinates = coordinates,
        eulerAnglesOfPlane = planeAngles,
        eulerAnglesOfCamera = cameraAngles
    )

    # Capture image
    dls = decklinksrc.DeckLinkSRC()
    imageBase = dls.grab()

    print("Click on the image and geolocation will be run with that location, exit with spacebar")
    locator = geolocation.Geolocation()
    locator.set_constants()

    cv2.namedWindow("geolocationImage")
    cv2.setMouseCallback("geolocationImage", locate)

    while (1):
        cv2.imshow("geolocationImage", imageBase)
        # Stop on spacebar
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.destroyAllWindows()
            break



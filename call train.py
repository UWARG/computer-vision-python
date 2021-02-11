import os

def _main_():
    if os.path.exists("targetAcquisition/yolov2_assets"):
        os.chdir("targetAcquisition/yolov2_assets")
"""
Changing directory to yolov2_assets to get config.json

"""
        from yolov2_assets import train
        train(config.json)
    else :
        print ("YOLOV2_ASSETS Directory not found. Specify path")
if _name_ == 'main':
    main()

import os

def _main_():
    main_directory=os.getcwd()
"""
stores current working directory prior to change

"""
    if os.path.exists("targetAcquisition/yolov2_assets"):
        os.chdir("targetAcquisition/yolov2_assets")
"""
Changing directory to yolov2_assets to get config.json

"""
        from yolov2_assets import train
        train(config.json)
        os.chdir(main_directory)
    else :
        print ("YOLOV2_ASSETS Directory not found. Specify path")
if _name_ == 'main':
    main()

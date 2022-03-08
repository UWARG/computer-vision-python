#bash

pip install -r requirements.txt
cd modules/targetAcquisition/Yolov5_DeepSort_Pytorch
git submodule update --init
cd ../../../
python test_main.py 
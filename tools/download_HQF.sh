source /root/miniconda3/bin/activate
conda activate evreal-tools
cd /root/autodl-tmp/data_eventcnn
mkdir HQF
cd HQF
gdown https://drive.google.com/uc?id=1NfWunORSvR-1qycaOnCefPYXIgPwU-Ic

unzip rosbags.zip
rm rosbags.zip

conda activate eventcnn
source /opt/ros/noetic/setup.bash

python /root/wj/event_cnn_minimal/events_contrast_maximization/tools/rosbag_to_h5.py /root/autodl-tmp/data_eventcnn/HQF --output_dir /root/autodl-tmp/data_eventcnn/HQF_H5 --event_topic /dvs/events --image_topic /dvs/image_raw


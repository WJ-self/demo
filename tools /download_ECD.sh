source /root/miniconda3/bin/activate
conda activate evreal-tools
cd /root/autodl-tmp/data_eventcnn
mkdir ECD
cd ECD
wget https://rpg.ifi.uzh.ch/datasets/davis/boxes_6dof.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/calibration.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/dynamic_6dof.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/office_zigzag.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/poster_6dof.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/shapes_6dof.bag
wget https://rpg.ifi.uzh.ch/datasets/davis/slider_depth.bag

conda activate eventcnn
source /opt/ros/noetic/setup.bash

python /root/wj/event_cnn_minimal/events_contrast_maximization/tools/rosbag_to_h5.py /root/autodl-tmp/data_eventcnn/ECD --output_dir /root/autodl-tmp/data_eventcnn/ECD_H5 --event_topic /dvs/events --image_topic /dvs/image_raw
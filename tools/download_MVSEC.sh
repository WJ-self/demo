source ~/miniconda3/bin/activate
conda activate evreal-tools
cd /root/autodl-tmp/data_eventcnn
mkdir MVSEC
cd MVSEC
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying1_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying2_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying3_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/indoor_flying/indoor_flying4_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day1_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_day/outdoor_day2_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night1_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night2_data.bag
wget http://visiondata.cis.upenn.edu/mvsec/outdoor_night/outdoor_night3_data.bag

conda activate eventcnn
source /opt/ros/noetic/setup.bash

python /root/wj/event_cnn_minimal/events_contrast_maximization/tools/rosbag_to_h5.py /root/autodl-tmp/data_eventcnn/MVSEC --output_dir /root/autodl-tmp/data_eventcnn/MVSEC_H5 --event_topic /dvs/events --image_topic /dvs/image_raw
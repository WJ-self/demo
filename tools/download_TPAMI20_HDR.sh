source ~/miniconda3/bin/activate
conda activate evreal-tools
cd /root/autodl-tmp/data_eventcnn
mkdir TPAMI20
cd TPAMI20
wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_selfie.zip
# wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_tunnel.zip
# wget https://rpg.ifi.uzh.ch/data/E2VID/datasets/TPAMI/events/hdr_sun.zip
unzip hdr_selfie.zip
# unzip hdr_tunnel.zip
# unzip hdr_sun.zip
rm hdr_selfie.zip
# rm hdr_tunnel.zip
# rm hdr_sun.zip

conda activate eventcnn
source /opt/ros/noetic/setup.bash
python /root/wj/event_cnn_minimal/events_contrast_maximization/tools/rosbag_to_h5.py /root/autodl-tmp/data_eventcnn/TPAMI20 --output_dir /root/autodl-tmp/data_eventcnn/TPAMI20_H5 --event_topic /dvs/events --image_topic /dvs/image_raw
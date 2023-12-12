import os

def write_file_paths(directory, output_file):
    with open(output_file, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)
                file.write(file_path + '\n')

# 替换为你想要扫描的目录
directory_to_scan = '/root/autodl-tmp/data_eventcnn/HQF_H5'

# 输出文件的路径
output_file_path = os.path.join(directory_to_scan, 'data_file.txt')

write_file_paths(directory_to_scan, output_file_path)

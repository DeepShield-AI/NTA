import os
import shutil
 
# 设置你的pcap文件所在的目录
pcap_directory = r'/home/liz/test/dataset/ISCX-VPN-NonVPN-2016/PCAPs/splitVPN2'
# 设置你想要移动文件到的目标目录
destination_directory = r'/home/liz/test/dataset/ISCX-VPN-NonVPN-2016/PCAPs/splitVPN2/tcp'
 
# 确保目标目录存在，如果不存在则创建
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)
 
# 遍历目录中的所有文件
for filename in os.listdir(pcap_directory):
    # 检查文件名是否包含'TCP'
    if 'TCP' in filename:
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(pcap_directory, filename)
        destination_file = os.path.join(destination_directory, filename)
 
        # 移动文件
        shutil.move(source_file, destination_file)
        print(f'Moved: {filename}')

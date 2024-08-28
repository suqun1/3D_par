###########面积插值##################
import time
import os
import numpy as np
from area import calculate_rectangle_area,  find_closest_interpolation, append_interpolation_to_annotations

# 文件夹路径
annotations_folder = r'D:\GitHub\-AI4CELLBIO-ECNU\yolov7-main\runs\detect\exp106\labels'
images_folder = r'D:\GitHub\-AI4CELLBIO-ECNU\yolov7-main\runs\detect\exp106'
output_folder = r'D:\GitHub\-AI4CELLBIO-ECNU\yolov7-main\runs\detect\exp106'

# 创建输出文件夹，如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# # 示例类别字典和插值数据
category_dict = {0: -60, 1: -30, 2: 0, 3: 30, 4: 60}  # 示例类别字典
x_values = [39153.85, 24335.66, 14520.12, 27059.76, 38611.89]  # 示例x值（原始面积值）
y_values = [-60, -30, 0, 30, 60]  # 示例y值
# x_values = [1936, 1296, 784, 399, 209,156, 305, 756, 1296, 1806, 2550]
# y_values = [-70, -60, -50, -40, -30,-25, -20, -10, 0, 10, 20]
# category_dict = {
#     0: -70, 1: -60, 2: -50, 3: -40, 4: -30, 5: -20, 6: -10,7: 0,8:10,9:20
# }
# x_values = [1936,1600, 1296,1024, 784,576, 399,306, 209,196, 305,506, 784,1024, 1296,1600, 1936, 2256,2601]
# y_values = [-70, -65,-60,-55, -50, -45,-40,-35, -30,-25, -20,-15, -10,-5, 0,5, 10, 15,20]
# category_dict = {
#     0: -70, 1: -65, 2: -60, 3: -55, 4: -50, 5: -45, 6: -40,7: -35,8:-30,9:-25,10:-20,
#     11:-15,12:-10,13:-5,14:0,15:5,16:10,17:15,18:20
# }
# 记录开始时间
start_time = time.time()
# 遍历文件夹中的所有标注文件
for filename in os.listdir(annotations_folder):
    if filename.endswith('.txt'):
        annotation_path = os.path.join(annotations_folder, filename)
        image_filename = filename.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_filename)

        # 确保图像文件存在
        if os.path.exists(image_path):
            # 计算矩形框的面积
            areas = calculate_rectangle_area(annotation_path, image_path)
            print("面积",areas)

            # 找到最接近类别值的插值结果
            closest_results = find_closest_interpolation(areas, category_dict, x_values, y_values, annotation_path)

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 将插值结果添加到YOLO标注文件中
            append_interpolation_to_annotations(annotation_path, closest_results, output_path)

            print(f"插值结果已添加到文件: {output_path}")
# 记录结束时间
end_time = time.time()
# 计算并输出总时间
total_time = end_time - start_time
print(f"代码执行总时间: {total_time:.2f} 秒")
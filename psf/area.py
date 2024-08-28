import numpy as np
from PIL import Image

def calculate_rectangle_area(file_path, image_path):
    """
    读取YOLO测试结果txt文件和对应的图像，并计算每个矩形框的面积。

    :param file_path: YOLO测试结果txt文件的路径
    :param image_path: 对应图像的路径
    :return: 矩形框的面积列表
    """
    areas = []

    # 加载图像并获取尺寸
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # 打开并读取txt文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()

        # 检查是否有足够的元素（至少5个：class_id, x_center, y_center, width, height）
        if len(parts) < 5:
            print(f"Skipping line due to insufficient data: {line.strip()}")
            continue

        try:
            # 解析每一行，提取坐标信息
            class_id = parts[0]
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            # 计算矩形框的左上角和右下角坐标
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

            # 计算矩形框的面积
            area = width * height
            area = round(area, 2)

            # 将面积添加到列表中
            areas.append(area)
        except ValueError as e:
            print(f"Skipping line due to value error: {line.strip()} - {e}")

    return areas


def linear_interpolate_complete(x_input, x_values, y_values):
    interpolation_results = []
    max_index = np.argmax(x_values)  # 找到最大值的索引

    # 首先处理x_input在x_values常规区间内的情况
    if x_input <= x_values[max_index]:
        # 遍历所有的x值区间
        for i in range(len(x_values) - 1):
            # 检查x_input是否在当前区间内
            if (x_values[i] <= x_input <= x_values[i + 1]) or (x_values[i] >= x_input >= x_values[i + 1]):
                slope = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
                y_interpolated = y_values[i] + slope * (x_input - x_values[i])
                interpolation_results.append(y_interpolated)
    else:
        # 处理x_input大于x_values中的最大值的情况
        if max_index - 1 >= 0:
            left_index = max_index - 1
            slope_left = (y_values[max_index] - y_values[left_index]) / (x_values[max_index] - x_values[left_index])
            y_left = y_values[left_index] + slope_left * (x_input - x_values[left_index])
            interpolation_results.append(y_left)

        if max_index + 1 < len(x_values):
            right_index = max_index + 1
            slope_right = (y_values[right_index] - y_values[max_index]) / (x_values[right_index] - x_values[max_index])
            y_right = y_values[max_index] + slope_right * (x_input - x_values[max_index])
            interpolation_results.append(y_right)

    return interpolation_results

def find_closest_interpolation(areas, category_dict, x_values, y_values, annotation_path):
    """
    对每个面积值进行插值计算，并找出最接近其类别值的插值结果。

    :param areas: 一系列面积值
    :param category_dict: 类别值的字典
    :param x_values: 插值的x值
    :param y_values: 插值的y值
    :param annotation_path: 标注文件的路径
    :return: 每个面积值对应的最接近类别值的插值结果列表
    """
    closest_results = []

    # 读取标注文件
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    for index, area in enumerate(areas):
        # 进行插值计算
        interpolation_results = linear_interpolate_complete(area, x_values, y_values)
        print("插值结果",interpolation_results)
        print("area",area)

        # 读取对应的类别索引
        parts = annotations[index].strip().split()
        category_index = int(parts[0])
        category_value = category_dict[category_index]

        # 初始化最小差值和对应的插值结果
        min_distance = float('inf')
        closest_interpolation = None

        # 寻找最接近的插值结果
        for y_interpolated in interpolation_results:
            distance = abs(y_interpolated - category_value)
            if distance < min_distance:
                min_distance = distance
                closest_interpolation = y_interpolated

        closest_results.append(closest_interpolation)

    return closest_results

def append_interpolation_to_annotations(annotation_path, interpolated_values, output_path):
    """
    将插值结果添加到YOLO标注文件中。

    :param annotation_path: 原始YOLO标注文件的路径
    :param interpolated_values: 插值结果列表
    :param output_path: 输出的标注文件路径
    """
    updated_annotations = []

    # 读取原始标注文件
    with open(annotation_path, 'r') as file:
        for i, line in enumerate(file):
            if i < len(interpolated_values):
                # 检查插值结果是否为 None，如果是，则使用默认值 0
                interpolated_value = interpolated_values[i] if interpolated_values[i] is not None else 0
                updated_line = line.strip() + ' ' + '{:.4f}'.format(interpolated_value) + '\n'
                updated_annotations.append(updated_line)
            else:
                # 如果没有对应的插值结果，保留原始标注
                updated_annotations.append(line)

    # 将更新后的标注写入新文件
    with open(output_path, 'w') as file:
        for line in updated_annotations:
            file.write(line)

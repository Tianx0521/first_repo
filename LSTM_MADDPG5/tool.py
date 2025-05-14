def polygon_area_and_centroid(points):
    """
    计算多边形的面积和中心点坐标。

    参数:
        points (list): 由点组成的列表，每个点为(x, y)元组。

    返回:
        tuple: 包含面积（绝对值）和中心点坐标的元组，格式为(area, (centroid_x, centroid_y))。

    异常:
        ValueError: 如果点的数量少于3个或点共线导致面积为0。
    """
    n = len(points)
    if n < 3:
        raise ValueError("Polygon must have at least 3 points")

    sum_area = 0.0
    sum_cx = 0.0
    sum_cy = 0.0

    for i in range(n):
        x_i, y_i = points[i]
        x_j, y_j = points[(i + 1) % n]  # 处理最后一个点连接到第一个点

        factor = x_i * y_j - x_j * y_i
        sum_area += factor
        sum_cx += (x_i + x_j) * factor
        sum_cy += (y_i + y_j) * factor

    area = abs(sum_area) / 2.0
    if area == 0:
        raise ValueError("The points are collinear, area is zero")

    # 使用有符号的sum_area计算质心
    centroid_x = sum_cx / (3 * sum_area)
    centroid_y = sum_cy / (3 * sum_area)

    return (area, (centroid_x, centroid_y))


# 示例用法
if __name__ == "__main__":
    # 测试正方形
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    area, centroid = polygon_area_and_centroid(square)
    print(f"Area: {area}, Centroid: {centroid}")  # Area: 16.0, Centroid: (2.0, 2.0)

    # 测试三角形
    triangle = [(0, 0), (2, 0), (0, 2)]
    area, centroid = polygon_area_and_centroid(triangle)
    print(f"Area: {area}, Centroid: {centroid}")  # Area: 2.0, Centroid: (0.666..., 0.666...)
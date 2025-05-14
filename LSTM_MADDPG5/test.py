import math

def sort_points_counterclockwise(points):
    """
    按照逆时针方向对二维点进行排序。

    参数:
    points (list of lists): 包含N个二维坐标的列表，例如 [[x1, y1], [x2, y2], ...]

    返回:
    list of lists: 按逆时针顺序排序后的点列表
    """
    if not points:
        return []

    # 找到基准点（最左下方的点）
    base_point = min(points, key=lambda p: (p[1], p[0]))

    def angle_from_base(point):
        """计算点相对于基准点的极角"""
        dx = point[0] - base_point[0]
        dy = point[1] - base_point[1]
        return math.atan2(dy, dx)

    # 移除基准点后，对其余点按极角排序
    other_points = [p for p in points if p != base_point]
    sorted_other = sorted(other_points, key=angle_from_base)

    # 将基准点放在首位
    sorted_points = [base_point] + sorted_other

    return sorted_points


points = [
    [2, 2],
    [2, 0],
    [4, 1],
    [0, 0],
    [4, 4],
    [1, 1],
    [2, -1]
]

sorted_points = sort_points_counterclockwise(points)
print(sorted_points)

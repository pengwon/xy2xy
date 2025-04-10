#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
坐标系线性变换工具

这个程序可以将一个坐标系中的点坐标转换到另一个坐标系中。
通过提供两个坐标系中各自的两个对应点，程序计算变换矩阵，
然后使用该矩阵将CSV文件中的坐标从坐标系1变换到坐标系2。
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os


def calculate_transformation(o, p):
    # 确保输入是numpy数组
    o = np.array(o, dtype=float)
    p = np.array(p, dtype=float)

    delta_x = p[1, 0] - p[0, 0] - o[0]
    delta_y = p[1, 1] - p[0, 1] - o[1]

    # 计算缩放因子
    kx = delta_x / p[0, 1] if p[0, 1] != 0 else 0
    ky = delta_y / p[0, 0] if p[0, 0] != 0 else 0

    return kx, ky


def transform_coordinates(x, y, kx, ky):
    """
    Transforms the coordinates (x, y) using the given scaling factors.

    Parameters:
        x (float): The x-coordinate to be transformed.
        y (float): The y-coordinate to be transformed.
        kx (float): The scaling factor for the x-coordinate.
        ky (float): The scaling factor for the y-coordinate.

    Returns:
        tuple: A tuple containing the transformed x and y coordinates (x', y').
    """
    point = np.array([x, y])
    transformed_x = point[0] + kx * point[1]
    transformed_y = point[1] + ky * point[0]
    return transformed_x, transformed_y


def transform_csv(
    input_file,
    output_file,
    kx,
    ky,
    x_col="Center-X(mm)",
    y_col="Center-Y(mm)",
    x_out_col="Center-X(mm)",
    y_out_col="Center-Y(mm)",
):
    """
    转换CSV文件中的坐标

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        transform_matrix: 变换矩阵
        translation_vector: 平移向量
        x_col, y_col: 输入CSV文件中X和Y坐标的列名
        x_out_col, y_out_col: 输出CSV文件中转换后坐标的列名
    """
    try:
        # 先读取文件内容，逐行检查以找到标题行
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 寻找可能包含所需列名的行
        header_row = 0
        for i, line in enumerate(lines):
            if x_col in line and y_col in line:
                header_row = i
                print(f"找到标题行: 第{header_row + 1}行")
                break

        # 尝试使用找到的标题行读取CSV
        try:
            df = pd.read_csv(input_file, skiprows=header_row)

            # 检查列是否确实存在于读取的DataFrame中
            if x_col not in df.columns or y_col not in df.columns:
                # 如果列名存在于行中但读取后不在DataFrame的列中，可能是因为分隔符问题
                # 尝试使用不同的分隔符
                for sep in [",", ";", "\t"]:
                    df = pd.read_csv(input_file, skiprows=header_row, sep=sep)
                    if x_col in df.columns and y_col in df.columns:
                        print(f"使用分隔符 '{sep}' 成功读取文件")
                        break

        except Exception as e:
            print(f"尝试读取标题行时出错: {e}")
            # 尝试跳过更多行，假设Altium文件格式
            print("尝试使用Altium标准格式读取...")
            df = pd.read_csv(input_file, skiprows=18)

        # 最终检查列是否存在
        if x_col not in df.columns or y_col not in df.columns:
            print(f"错误: 无法在CSV文件中找到列 '{x_col}' 或 '{y_col}'")
            print(f"可用的列: {', '.join(df.columns)}")
            return False

        # 应用变换
        transformed_coords = np.array(
            [
                transform_coordinates(row[x_col], row[y_col], kx, ky)
                for _, row in df.iterrows()
            ]
        )

        # 添加转换后的坐标到DataFrame，保留4位小数
        df[x_out_col] = np.round(transformed_coords[:, 0], 4)
        df[y_out_col] = np.round(transformed_coords[:, 1], 4)

        # 保存结果
        df.to_csv(output_file, index=False)
        return True

    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback

        print(traceback.format_exc())
        return False


def parse_point(point_str):
    """解析点坐标字符串，格式为'x,y'"""
    try:
        x, y = map(float, point_str.split(","))
        return [x, y]
    except ValueError:
        raise argparse.ArgumentTypeError("点坐标格式必须为'x,y'（例如: '1.0,2.5'）")


def main():
    parser = argparse.ArgumentParser(description="坐标系线性变换工具")
    parser.add_argument(
        "--o",
        type=parse_point,
        required=True,
        help="坐标系2中对应原点的坐标，格式为'x,y'",
    )
    parser.add_argument(
        "--p1",
        type=parse_point,
        required=True,
        help="坐标系1中的第一个点，格式为'x,y'",
    )
    parser.add_argument(
        "--p2",
        type=parse_point,
        required=True,
        help="坐标系2中对应的第一个点，格式为'x,y'",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="输入CSV文件路径"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="输出CSV文件路径，默认为'<输入文件名>_transformed.csv'",
    )
    parser.add_argument(
        "--xcol",
        type=str,
        default="Center-X(mm)",
        help="输入CSV中X坐标的列名，默认为'Center-X(mm)'",
    )
    parser.add_argument(
        "--ycol",
        type=str,
        default="Center-Y(mm)",
        help="输入CSV中Y坐标的列名，默认为'Center-Y(mm)'",
    )
    parser.add_argument(
        "--xcol-out",
        type=str,
        default="Center-X(mm)",
        help="输出CSV中转换后X坐标的列名，默认为'Center-X(mm)'",
    )
    parser.add_argument(
        "--ycol-out",
        type=str,
        default="Center-Y(mm)",
        help="输出CSV中转换后Y坐标的列名，默认为'Center-Y(mm)'",
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        return 1

    # 如果未指定输出文件，生成默认名称
    if args.output is None:
        filename, ext = os.path.splitext(args.input)
        args.output = f"{filename}_transformed{ext}"

    # 计算变换矩阵
    o = args.o
    p = [args.p1, args.p2]

    # 打印输入信息
    print("坐标变换信息:")
    print(f"坐标系2中原点: {o}")
    print(f"坐标系1中的点: {p[0]}")
    print(f"坐标系2中对应的点: {p[1]}")

    kx, ky = calculate_transformation(o, p)

    # 打印变换矩阵
    print("\n计算得到的变换参数:")
    print("kx, ky:", kx, ky)

    # 转换CSV文件
    print(f"\n转换文件 '{args.input}' 中的坐标...")
    success = transform_csv(
        args.input,
        args.output,
        kx,
        ky,
        args.xcol,
        args.ycol,
        args.xcol_out,
        args.ycol_out,
    )

    if success:
        print(f"转换完成，结果已保存至 '{args.output}'")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

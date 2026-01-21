"""
翻转 PKL 文件中的 faces 顶点顺序以反转法线方向。

用法:
    python flip_faces.py <input_pkl> [output_pkl]

如果不指定 output_pkl，将覆盖原文件。

示例:
    python flip_faces.py body.pkl body_flipped.pkl
    python flip_faces.py cloth_template.pkl
"""

import sys
import os
import argparse
import pickle
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def flip_faces(faces: np.ndarray) -> np.ndarray:
    """
    翻转 faces 的顶点顺序以反转法线方向。
    
    原理：三角形法线由顶点顺序决定（右手定则）
    [v0, v1, v2] -> [v0, v2, v1] 会反转法线方向
    
    Args:
        faces: [F, 3] 三角形面片数组
        
    Returns:
        翻转后的 faces 数组
    """
    faces = np.array(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces 形状应为 [F, 3]，当前为 {faces.shape}")
    
    # 交换第 1 和第 2 个顶点索引
    flipped = faces[:, [0, 2, 1]]
    return flipped


def process_pkl(input_path: str, output_path: str, face_keys: list = None):
    """
    处理 PKL 文件，翻转其中的 faces。
    
    Args:
        input_path: 输入 PKL 文件路径
        output_path: 输出 PKL 文件路径
        face_keys: 要翻转的 faces 字段名列表，默认为 ['faces']
    """
    if face_keys is None:
        face_keys = ['faces']
    
    print(f"读取: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    flipped_count = 0
    
    # 处理字典类型数据
    if isinstance(data, dict):
        for key in face_keys:
            if key in data:
                original_shape = np.array(data[key]).shape
                data[key] = flip_faces(data[key])
                print(f"  ✓ 已翻转 '{key}': {original_shape}")
                flipped_count += 1
            else:
                print(f"  - 未找到 '{key}' 字段")
        
        # 显示数据中的所有键
        print(f"\n数据包含的字段: {list(data.keys())}")
    else:
        print(f"警告: PKL 文件不是字典类型，而是 {type(data)}")
        return
    
    if flipped_count == 0:
        print("\n警告: 没有翻转任何 faces 字段！")
        return
    
    # 保存
    print(f"\n保存到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"完成！共翻转 {flipped_count} 个 faces 字段")


def main():
    parser = argparse.ArgumentParser(
        description="翻转 PKL 文件中的 faces 顶点顺序以反转法线方向",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python flip_faces.py body.pkl body_flipped.pkl
    python flip_faces.py cloth_template.pkl
    python flip_faces.py data.pkl --keys faces body_faces
        """
    )
    
    parser.add_argument('input', help='输入 PKL 文件路径')
    parser.add_argument('output', nargs='?', default=None, 
                        help='输出 PKL 文件路径（默认覆盖原文件）')
    parser.add_argument('--keys', nargs='+', default=['faces'],
                        help='要翻转的 faces 字段名（默认: faces）')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output if args.output else input_path
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)
    
    # 如果覆盖原文件，先确认
    if output_path == input_path:
        print(f"警告: 将覆盖原文件 {input_path}")
        confirm = input("确认? (y/N): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            sys.exit(0)
    
    process_pkl(input_path, output_path, args.keys)


if __name__ == '__main__':
    main()


"""
pkl_to_hdf5.py
pkl 和 hdf5 格式动画序列相互转换

支持的 pkl 输入格式：
1. mesh_sequence: {'verts': [N, V, 3], 'faces': [F, 3]}
2. 新 mesh_sequence: {'vertices': [N, V, 3], 'faces': [F, 3]}
3. 推理输出: {'pred': [N, V, 3], 'cloth_faces': [F, 3], 'obstacle': ..., 'obstacle_faces': ...}
4. SMPL 参数: {'body_pose': [N, 69], 'global_orient': [N, 3], 'transl': [N, 3], 'betas': [10,]}

HDF5 结构：
- Mesh 格式:
    /vertices          [N, V, 3] float32
    /faces             [F, 3] int64
    /obstacle_vertices [N, V, 3] float32 (可选)
    /obstacle_faces    [F, 3] int64 (可选)
    attrs: format, version, num_frames, num_vertices, num_faces, source_path

- SMPL 格式:
    /body_pose         [N, 69] float32
    /global_orient     [N, 3] float32
    /transl            [N, 3] float32
    /betas             [10,] 或 [N, 10] float32
    attrs: format, version, num_frames, source_path

使用示例：
    # pkl -> hdf5
    python pkl_to_hdf5.py input.pkl
    python pkl_to_hdf5.py input.pkl -o output.h5
    python pkl_to_hdf5.py --batch input_dir/ --output-dir output_dir/
    python pkl_to_hdf5.py input.pkl --compress
    
    # hdf5 -> pkl
    python pkl_to_hdf5.py --to-pkl input.h5
    python pkl_to_hdf5.py --to-pkl input.h5 -o output.pkl
    python pkl_to_hdf5.py --to-pkl-batch input_dir/ --output-dir output_dir/
    
    # 查看 hdf5 文件信息
    python pkl_to_hdf5.py --info file.h5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

# 设置项目路径 (Tests/dataProcess -> Tests -> HOOD)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import pickle_load, pickle_dump

# 检查 h5py 是否安装
try:
    import h5py
except ImportError:
    print("错误: 未安装 h5py 库")
    print("请运行: pip install h5py")
    sys.exit(1)


# ============ 常量定义 ============

class HDF5Format:
    """HDF5 格式常量"""
    VERSION = "1.0"
    MESH_FORMAT = "mesh_sequence_hdf5"
    SMPL_FORMAT = "smpl_sequence_hdf5"
    
    # 压缩配置
    COMPRESSION = "gzip"
    COMPRESSION_OPTS = 4  # 压缩级别 1-9


class SourceFormat:
    """源格式类型"""
    MESH = "mesh"
    MESH_NEW = "mesh_new"  # vertices 字段格式
    INFERENCE = "inference"
    SMPL = "smpl"


# ============ 格式检测 ============

def detect_source_format(data: Dict[str, Any]) -> str:
    """
    检测 pkl 数据的格式类型
    
    :param data: 加载的 pkl 数据字典
    :return: 格式类型字符串
    """
    keys = set(data.keys())
    
    # SMPL 格式检测
    smpl_keys = {'body_pose', 'global_orient', 'transl', 'betas'}
    if smpl_keys.issubset(keys):
        return SourceFormat.SMPL
    
    # 推理输出格式
    if 'pred' in keys and 'cloth_faces' in keys:
        return SourceFormat.INFERENCE
    
    # 新 mesh 格式 (vertices)
    if 'vertices' in keys and 'faces' in keys:
        return SourceFormat.MESH_NEW
    
    # 标准 mesh 格式 (verts)
    if 'verts' in keys and 'faces' in keys:
        return SourceFormat.MESH
    
    raise ValueError(f"无法识别的 pkl 格式，包含的键: {list(keys)}")


# ============ 转换函数 ============

def convert_mesh_to_hdf5(
    data: Dict[str, Any],
    output_path: str,
    source_path: str = "",
    compress: bool = False,
    source_format: str = SourceFormat.MESH
) -> None:
    """
    将 Mesh 格式的 pkl 数据转换为 HDF5
    
    :param data: pkl 数据字典
    :param output_path: 输出 HDF5 文件路径
    :param source_path: 源文件路径（用于元数据）
    :param compress: 是否启用压缩
    :param source_format: 源数据格式类型
    """
    # 提取顶点和面数据
    if source_format == SourceFormat.INFERENCE:
        vertices = np.array(data['pred'], dtype=np.float32)
        faces = np.array(data['cloth_faces'], dtype=np.int64)
        if len(faces.shape) == 3:
            faces = faces[0]
        
        # 处理障碍物
        obstacle_vertices = None
        obstacle_faces = None
        if 'obstacle' in data and 'obstacle_faces' in data:
            obstacle_vertices = np.array(data['obstacle'], dtype=np.float32)
            obstacle_faces = np.array(data['obstacle_faces'], dtype=np.int64)
            if len(obstacle_faces.shape) == 3:
                obstacle_faces = obstacle_faces[0]
    
    elif source_format == SourceFormat.MESH_NEW:
        vertices = np.array(data['vertices'], dtype=np.float32)
        faces = np.array(data['faces'], dtype=np.int64)
        obstacle_vertices = None
        obstacle_faces = None
        
        if 'obstacle_vertices' in data:
            obstacle_vertices = np.array(data['obstacle_vertices'], dtype=np.float32)
        if 'obstacle_faces' in data:
            obstacle_faces = np.array(data['obstacle_faces'], dtype=np.int64)
    
    else:  # SourceFormat.MESH
        vertices = np.array(data['verts'], dtype=np.float32)
        faces = np.array(data['faces'], dtype=np.int64)
        obstacle_vertices = None
        obstacle_faces = None
    
    # 确保顶点是 3D 张量 [N, V, 3]
    if len(vertices.shape) == 2:
        vertices = vertices[np.newaxis, ...]
    
    # 压缩配置
    compression_kwargs = {}
    if compress:
        compression_kwargs = {
            'compression': HDF5Format.COMPRESSION,
            'compression_opts': HDF5Format.COMPRESSION_OPTS
        }
    
    # 写入 HDF5 文件
    with h5py.File(output_path, 'w') as f:
        # 主数据集
        f.create_dataset('vertices', data=vertices, **compression_kwargs)
        f.create_dataset('faces', data=faces, **compression_kwargs)
        
        # 障碍物数据集（如果存在）
        if obstacle_vertices is not None:
            if len(obstacle_vertices.shape) == 2:
                obstacle_vertices = obstacle_vertices[np.newaxis, ...]
            f.create_dataset('obstacle_vertices', data=obstacle_vertices, **compression_kwargs)
        
        if obstacle_faces is not None:
            f.create_dataset('obstacle_faces', data=obstacle_faces, **compression_kwargs)
        
        # 元数据属性
        f.attrs['format'] = HDF5Format.MESH_FORMAT
        f.attrs['version'] = HDF5Format.VERSION
        f.attrs['num_frames'] = vertices.shape[0]
        f.attrs['num_vertices'] = vertices.shape[1]
        f.attrs['num_faces'] = faces.shape[0]
        f.attrs['source_path'] = source_path
        f.attrs['source_format'] = source_format
        f.attrs['compressed'] = compress
    
    print(f"  已保存: {output_path}")
    print(f"    - 帧数: {vertices.shape[0]}")
    print(f"    - 顶点数: {vertices.shape[1]}")
    print(f"    - 面数: {faces.shape[0]}")
    if obstacle_vertices is not None:
        print(f"    - 障碍物: 是")


def convert_smpl_to_hdf5(
    data: Dict[str, Any],
    output_path: str,
    source_path: str = "",
    compress: bool = False
) -> None:
    """
    将 SMPL 格式的 pkl 数据转换为 HDF5
    
    :param data: pkl 数据字典
    :param output_path: 输出 HDF5 文件路径
    :param source_path: 源文件路径（用于元数据）
    :param compress: 是否启用压缩
    """
    # 提取 SMPL 参数
    body_pose = np.array(data['body_pose'], dtype=np.float32)
    global_orient = np.array(data['global_orient'], dtype=np.float32)
    transl = np.array(data['transl'], dtype=np.float32)
    betas = np.array(data['betas'], dtype=np.float32)
    
    num_frames = body_pose.shape[0]
    
    # 压缩配置
    compression_kwargs = {}
    if compress:
        compression_kwargs = {
            'compression': HDF5Format.COMPRESSION,
            'compression_opts': HDF5Format.COMPRESSION_OPTS
        }
    
    # 写入 HDF5 文件
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('body_pose', data=body_pose, **compression_kwargs)
        f.create_dataset('global_orient', data=global_orient, **compression_kwargs)
        f.create_dataset('transl', data=transl, **compression_kwargs)
        f.create_dataset('betas', data=betas, **compression_kwargs)
        
        # 元数据属性
        f.attrs['format'] = HDF5Format.SMPL_FORMAT
        f.attrs['version'] = HDF5Format.VERSION
        f.attrs['num_frames'] = num_frames
        f.attrs['source_path'] = source_path
        f.attrs['compressed'] = compress
    
    print(f"  已保存: {output_path}")
    print(f"    - 帧数: {num_frames}")
    print(f"    - betas 形状: {betas.shape}")


def convert_pkl_to_hdf5(
    input_path: str,
    output_path: Optional[str] = None,
    compress: bool = False
) -> str:
    """
    将 pkl 文件转换为 HDF5 格式
    
    :param input_path: 输入 pkl 文件路径
    :param output_path: 输出 HDF5 文件路径（默认与输入同名，后缀改为 .h5）
    :param compress: 是否启用压缩
    :return: 输出文件路径
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if not input_path.suffix.lower() == '.pkl':
        raise ValueError(f"输入文件必须是 .pkl 格式: {input_path}")
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path.with_suffix('.h5')
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"转换: {input_path}")
    
    # 加载 pkl 数据
    data = pickle_load(str(input_path))
    
    # 检测格式
    source_format = detect_source_format(data)
    print(f"  检测到格式: {source_format}")
    
    # 根据格式转换
    if source_format == SourceFormat.SMPL:
        convert_smpl_to_hdf5(
            data, 
            str(output_path), 
            source_path=str(input_path),
            compress=compress
        )
    else:
        convert_mesh_to_hdf5(
            data,
            str(output_path),
            source_path=str(input_path),
            compress=compress,
            source_format=source_format
        )
    
    return str(output_path)


def batch_convert(
    input_dir: str,
    output_dir: Optional[str] = None,
    compress: bool = False,
    recursive: bool = False
) -> Tuple[int, int]:
    """
    批量转换目录下的所有 pkl 文件
    
    :param input_dir: 输入目录
    :param output_dir: 输出目录（默认与输入目录相同）
    :param compress: 是否启用压缩
    :param recursive: 是否递归处理子目录
    :return: (成功数, 失败数)
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 pkl 文件
    if recursive:
        pkl_files = list(input_dir.rglob("*.pkl"))
    else:
        pkl_files = list(input_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"未找到 pkl 文件: {input_dir}")
        return 0, 0
    
    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    print("=" * 50)
    
    success_count = 0
    fail_count = 0
    
    for pkl_file in pkl_files:
        try:
            # 计算相对路径以保持目录结构
            rel_path = pkl_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.h5')
            
            convert_pkl_to_hdf5(
                str(pkl_file),
                str(output_path),
                compress=compress
            )
            success_count += 1
        except Exception as e:
            print(f"  转换失败: {pkl_file}")
            print(f"    错误: {e}")
            fail_count += 1
    
    print("=" * 50)
    print(f"转换完成: 成功 {success_count}, 失败 {fail_count}")
    
    return success_count, fail_count


# ============ HDF5 -> PKL 转换 ============

def convert_hdf5_to_pkl(
    input_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    将 HDF5 文件转换为 pkl 格式
    
    :param input_path: 输入 HDF5 文件路径
    :param output_path: 输出 pkl 文件路径（默认与输入同名，后缀改为 .pkl）
    :return: 输出文件路径
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if input_path.suffix.lower() not in ['.h5', '.hdf5']:
        raise ValueError(f"输入文件必须是 .h5 或 .hdf5 格式: {input_path}")
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path.with_suffix('.pkl')
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"转换: {input_path}")
    
    # 读取 HDF5 数据
    with h5py.File(str(input_path), 'r') as f:
        # 检测格式
        file_format = f.attrs.get('format', '').decode() if isinstance(f.attrs.get('format', ''), bytes) else f.attrs.get('format', '')
        
        data = {}
        
        if file_format == HDF5Format.SMPL_FORMAT:
            # SMPL 格式
            data['body_pose'] = np.array(f['body_pose'])
            data['global_orient'] = np.array(f['global_orient'])
            data['transl'] = np.array(f['transl'])
            data['betas'] = np.array(f['betas'])
            print(f"  检测到格式: SMPL")
            print(f"  已保存: {output_path}")
            print(f"    - 帧数: {data['body_pose'].shape[0]}")
            
        else:
            # Mesh 格式（默认）
            data['vertices'] = np.array(f['vertices'])
            data['faces'] = np.array(f['faces'])
            
            # 障碍物数据（如果存在）
            if 'obstacle_vertices' in f:
                data['obstacle_vertices'] = np.array(f['obstacle_vertices'])
            if 'obstacle_faces' in f:
                data['obstacle_faces'] = np.array(f['obstacle_faces'])
            
            print(f"  检测到格式: Mesh")
            print(f"  已保存: {output_path}")
            print(f"    - 帧数: {data['vertices'].shape[0]}")
            print(f"    - 顶点数: {data['vertices'].shape[1]}")
            print(f"    - 面数: {data['faces'].shape[0]}")
            if 'obstacle_vertices' in data:
                print(f"    - 障碍物: 是")
    
    # 保存为 pkl
    pickle_dump(data, str(output_path))
    
    return str(output_path)


def batch_convert_hdf5_to_pkl(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False
) -> Tuple[int, int]:
    """
    批量将目录下的所有 HDF5 文件转换为 pkl 格式
    
    :param input_dir: 输入目录
    :param output_dir: 输出目录（默认与输入目录相同）
    :param recursive: 是否递归处理子目录
    :return: (成功数, 失败数)
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 hdf5 文件
    if recursive:
        h5_files = list(input_dir.rglob("*.h5")) + list(input_dir.rglob("*.hdf5"))
    else:
        h5_files = list(input_dir.glob("*.h5")) + list(input_dir.glob("*.hdf5"))
    
    if not h5_files:
        print(f"未找到 HDF5 文件: {input_dir}")
        return 0, 0
    
    print(f"找到 {len(h5_files)} 个 HDF5 文件")
    print("=" * 50)
    
    success_count = 0
    fail_count = 0
    
    for h5_file in h5_files:
        try:
            # 计算相对路径以保持目录结构
            rel_path = h5_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.pkl')
            
            convert_hdf5_to_pkl(str(h5_file), str(output_path))
            success_count += 1
        except Exception as e:
            print(f"  转换失败: {h5_file}")
            print(f"    错误: {e}")
            fail_count += 1
    
    print("=" * 50)
    print(f"转换完成: 成功 {success_count}, 失败 {fail_count}")
    
    return success_count, fail_count


# ============ HDF5 文件信息查看 ============

def show_hdf5_info(file_path: str) -> None:
    """
    显示 HDF5 文件的详细信息
    
    :param file_path: HDF5 文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"文件: {file_path}")
    print("=" * 50)
    
    with h5py.File(str(file_path), 'r') as f:
        # 显示属性
        print("属性:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\n数据集:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}:")
            print(f"    - 形状: {dataset.shape}")
            print(f"    - 类型: {dataset.dtype}")
            if dataset.compression:
                print(f"    - 压缩: {dataset.compression}")


# ============ 命令行接口 ============

def parse_args():
    parser = argparse.ArgumentParser(
        description='pkl 和 hdf5 格式动画序列相互转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # pkl -> hdf5: 转换单个文件
  python pkl_to_hdf5.py input.pkl
  python pkl_to_hdf5.py input.pkl -o output.h5
  python pkl_to_hdf5.py input.pkl --compress
  
  # pkl -> hdf5: 批量转换
  python pkl_to_hdf5.py --batch input_dir/
  python pkl_to_hdf5.py --batch input_dir/ --output-dir output_dir/
  python pkl_to_hdf5.py --batch input_dir/ --recursive
  
  # hdf5 -> pkl: 转换单个文件
  python pkl_to_hdf5.py --to-pkl input.h5
  python pkl_to_hdf5.py --to-pkl input.h5 -o output.pkl
  
  # hdf5 -> pkl: 批量转换
  python pkl_to_hdf5.py --to-pkl-batch input_dir/
  python pkl_to_hdf5.py --to-pkl-batch input_dir/ --output-dir output_dir/
  
  # 查看 HDF5 文件信息
  python pkl_to_hdf5.py --info file.h5
"""
    )
    
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        default=None,
        help='输入 pkl 文件路径 (pkl -> hdf5)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径（默认与输入同名，后缀自动更改）'
    )
    parser.add_argument(
        '--compress', '-c',
        action='store_true',
        help='启用 gzip 压缩 (仅 pkl -> hdf5)'
    )
    parser.add_argument(
        '--batch', '-b',
        type=str,
        default=None,
        metavar='DIR',
        help='批量转换目录下的所有 pkl 文件 (pkl -> hdf5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='批量转换的输出目录'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归处理子目录'
    )
    parser.add_argument(
        '--info', '-i',
        type=str,
        default=None,
        metavar='H5_FILE',
        help='显示 HDF5 文件信息'
    )
    
    # hdf5 -> pkl 参数
    parser.add_argument(
        '--to-pkl',
        type=str,
        default=None,
        metavar='H5_FILE',
        help='将 HDF5 文件转换为 pkl 格式'
    )
    parser.add_argument(
        '--to-pkl-batch',
        type=str,
        default=None,
        metavar='DIR',
        help='批量将目录下的 HDF5 文件转换为 pkl 格式'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 查看 HDF5 文件信息模式
    if args.info:
        show_hdf5_info(args.info)
        return
    
    # hdf5 -> pkl: 单文件转换
    if args.to_pkl:
        convert_hdf5_to_pkl(args.to_pkl, args.output)
        return
    
    # hdf5 -> pkl: 批量转换
    if args.to_pkl_batch:
        batch_convert_hdf5_to_pkl(
            args.to_pkl_batch,
            args.output_dir,
            recursive=args.recursive
        )
        return
    
    # pkl -> hdf5: 批量转换
    if args.batch:
        batch_convert(
            args.batch,
            args.output_dir,
            compress=args.compress,
            recursive=args.recursive
        )
        return
    
    # pkl -> hdf5: 单文件转换
    if args.input is None:
        print("错误: 请提供输入文件路径")
        print("  pkl -> hdf5: python pkl_to_hdf5.py input.pkl")
        print("  hdf5 -> pkl: python pkl_to_hdf5.py --to-pkl input.h5")
        print("使用 --help 查看帮助")
        sys.exit(1)
    
    convert_pkl_to_hdf5(
        args.input,
        args.output,
        compress=args.compress
    )


if __name__ == '__main__':
    main()


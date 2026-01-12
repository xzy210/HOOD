import h5py

def explore_hdf5(filename):
    """全面探索HDF5文件结构"""
    with h5py.File(filename, 'r') as f:
        def explore_group(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
                print(f"  大小: {obj.size}")
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        
        print(f"{'='*50}")
        print(f"文件: {filename}")
        print(f"{'='*50}")
        f.visititems(explore_group)

# 使用示例
explore_hdf5('E:\Projects4\HOOD\hood_data\EcoData\SK_TrainingSkirt_Struct_20260114_161713.hdf5')
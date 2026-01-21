import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
from utils.common import NodeType
from utils.cloth_and_material import load_obj
from utils.mesh_creation import add_coarse_edges

# 1. 从 OBJ 加载模板
vertices, faces = load_obj('hood_data/EcoData/split_output/surface_LayeredClothing_A0F407_ref.obj')

# 2. 创建模板字典
template_dict = {
    'vertices': vertices,
    'rest_pos': vertices,
    'faces': faces.astype(np.int64),
    'node_type': np.zeros((len(vertices), 1), dtype=np.int64)  # 默认全部为 0 (NORMAL)
}

# 3. 设置 pinned 顶点 (vertex_type = 3 = HANDLE)
pinned_indices = [0,1,12,18,24,30,36,42,48,54,60,66]  # 你需要固定的顶点索引
template_dict['node_type'][pinned_indices] = NodeType.HANDLE  # NodeType.HANDLE = 3

# 4. 添加 coarse edges
template_dict = add_coarse_edges(template_dict, n_levels=3)

# 5. 保存为 PKL
with open('hood_data/EcoData/split_output/surface_LayeredClothing_A0F407_ref.pkl', 'wb') as f:
    pickle.dump(template_dict, f)
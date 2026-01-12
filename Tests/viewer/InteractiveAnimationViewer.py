"""
InteractiveAnimationViewer.py
基于 aitviewer 的交互式动画可视化应用程序

功能：
1. 应用打开后通过 GUI 导入动画文件（支持多选）
2. 支持 pkl、hdf5、obj 格式
3. 同时播放多个动画，帧数取所有动画的最大值
4. 动画帧数提前结束时停留在最后一帧
5. OBJ 静态网格始终显示第一帧
6. 支持对导入的动画做平移操作

使用方法：
    python InteractiveAnimationViewer.py
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import imgui
import numpy as np

# 尝试导入 PyQt5（aitviewer 默认使用 PyQt5 窗口）
try:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QApplication, QFileDialog
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False
    # 回退到 tkinter
    from tkinter import Tk, filedialog as tk_filedialog

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent.parent  # HOOD project root
sys.path.insert(0, str(PROJECT_ROOT))

# Windows DLL 加载顺序问题修复
from aitviewer.headless import HeadlessRenderer  # noqa: F401
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from matplotlib import pyplot as plt

# 导入本地模块（处理不同运行方式的路径问题）
try:
    from AnimationViewer import (
        SequenceData,
        SequenceLoaderFactory,
        ViewerConfig,
        adjust_color,
        HDF5SequenceLoader,
    )
except ModuleNotFoundError:
    from Tests.AnimationViewer import (
        SequenceData,
        SequenceLoaderFactory,
        ViewerConfig,
        adjust_color,
        HDF5SequenceLoader,
    )


# ============ 动画项数据类 ============

@dataclass
class AnimationItem:
    """单个动画项的数据"""
    name: str                          # 显示名称
    data: SequenceData                 # 序列数据
    meshes: List[Any] = field(default_factory=list)  # 渲染对象（Meshes 或 Lines）
    original_vertices: np.ndarray = None  # 原始顶点位置（用于平移）
    obstacle_original_vertices: np.ndarray = None  # 障碍物原始顶点
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # 平移量
    color: Tuple[float, ...] = (0.5, 0.5, 0.5, 1.0)  # 颜色
    visible: bool = True               # 是否可见
    
    @property
    def n_frames(self) -> int:
        return self.data.n_frames
    
    @property
    def is_static(self) -> bool:
        return self.data.is_static
    
    @property
    def is_skeleton(self) -> bool:
        return self.data.is_skeleton


# ============ 交互式动画查看器 ============

class InteractiveAnimationViewer(Viewer):
    """
    交互式动画查看器
    
    继承自 aitviewer.Viewer，添加自定义 GUI 和动画管理功能
    """
    
    def __init__(self, **kwargs):
        # 设置默认窗口标题
        kwargs.setdefault('title', 'Interactive Animation Viewer')
        super().__init__(**kwargs)
        
        # 动画管理
        self.animation_items: List[AnimationItem] = []
        self.max_frames: int = 1
        self.color_index: int = 0
        self.cmap = plt.get_cmap('tab10')
        
        # 配置
        self.config = ViewerConfig()
        self.backface_culling = False
        
        # GUI 状态
        self._show_animation_panel = True
        self._selected_item_index = -1
        
        # 文件对话框状态
        self._file_dialog_helper = None
        
        # 注册自定义 GUI - 修改 gui_controls 字典
        self.gui_controls["menu"] = self._custom_gui_menu

        # 修复：在 Windows + QGLWidget(alpha buffer) 下，弹出对话框时可能出现“窗口透明/穿透”
        # 让 Qt 将该 OpenGL 窗口视为不透明绘制，避免 DWM 用 alpha 通道做合成。
        if HAS_PYQT5 and hasattr(self.wnd, "_widget"):
            try:
                self.wnd._widget.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
                self.wnd._widget.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
            except Exception as e:
                print(f"警告: 设置 Qt 窗口不透明属性失败: {e}")
        
        # 保存原始的 key_event 处理器
        self._original_key_event = self.key_event
    
    def key_event(self, key, action, modifiers):
        """处理键盘事件，添加自定义快捷键"""
        # Ctrl+I: 导入文件
        if key == self.wnd.keys.I and action == self.wnd.keys.ACTION_PRESS:
            if modifiers.ctrl:
                if HAS_PYQT5:
                    QtCore.QTimer.singleShot(0, self._open_import_dialog)
                else:
                    self._open_import_dialog()
                return
        
        # 调用原始的键盘事件处理
        super().key_event(key, action, modifiers)
    
    def _get_next_color(self) -> Tuple[float, ...]:
        """获取下一个颜色（循环使用色彩映射）"""
        color = self.cmap(self.color_index % 10)
        self.color_index += 1
        return color
    
    def _open_file_dialog(self, title: str = "选择文件", multiple: bool = True) -> List[str]:
        """
        打开文件选择对话框
        
        使用与 aitviewer 相同的 GUI 框架（PyQt5）来避免冲突
        
        :param title: 对话框标题
        :param multiple: 是否允许多选
        :return: 选择的文件路径列表
        """
        file_filter = "所有支持的格式 (*.pkl *.h5 *.hdf5 *.obj);;Pickle 文件 (*.pkl);;HDF5 文件 (*.h5 *.hdf5);;OBJ 文件 (*.obj);;所有文件 (*.*)"
        
        if HAS_PYQT5:
            return self._open_file_dialog_pyqt5(title, file_filter, multiple)
        else:
            return self._open_file_dialog_tkinter(title, multiple)
    
    def _open_file_dialog_pyqt5(self, title: str, file_filter: str, multiple: bool) -> List[str]:
        """使用 PyQt5 的文件对话框"""
        try:
            # 获取现有的 QApplication 实例（由 aitviewer 创建）
            app = QApplication.instance()
            if app is None:
                print("警告: 未找到 QApplication 实例")
                return []

            parent = None
            # 关键：将对话框挂到 viewer 的 Qt OpenGL widget 上，避免窗口层级/合成异常导致“透明”
            if hasattr(self.wnd, "_widget"):
                parent = self.wnd._widget

            # 避免某些 Windows 原生对话框与 QGLWidget 叠加时的合成问题
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            
            if multiple:
                files, _ = QFileDialog.getOpenFileNames(
                    parent,
                    title,
                    "",
                    file_filter,
                    options=options,
                )
                return list(files) if files else []
            else:
                file, _ = QFileDialog.getOpenFileName(
                    parent,
                    title,
                    "",
                    file_filter,
                    options=options,
                )
                return [file] if file else []
                
        except Exception as e:
            print(f"PyQt5 文件对话框错误: {e}")
            return []
    
    def _open_file_dialog_tkinter(self, title: str, multiple: bool) -> List[str]:
        """回退到 tkinter 文件对话框"""
        try:
            if self._file_dialog_helper is None:
                self._file_dialog_helper = Tk()
                self._file_dialog_helper.withdraw()
            
            filetypes = [
                ("所有支持的格式", "*.pkl *.h5 *.hdf5 *.obj"),
                ("Pickle 文件", "*.pkl"),
                ("HDF5 文件", "*.h5 *.hdf5"),
                ("OBJ 文件", "*.obj"),
                ("所有文件", "*.*")
            ]
            
            self._file_dialog_helper.update()
            
            if multiple:
                files = tk_filedialog.askopenfilenames(
                    title=title,
                    filetypes=filetypes,
                    parent=self._file_dialog_helper
                )
                result = list(files) if files else []
            else:
                file = tk_filedialog.askopenfilename(
                    title=title,
                    filetypes=filetypes,
                    parent=self._file_dialog_helper
                )
                result = [file] if file else []
            
            self._file_dialog_helper.update()
            return result
            
        except Exception as e:
            print(f"tkinter 文件对话框错误: {e}")
            return []
    
    def import_files(self, file_paths: List[str]) -> int:
        """
        导入动画文件
        
        :param file_paths: 文件路径列表
        :return: 成功导入的数量
        """
        success_count = 0
        
        for path in file_paths:
            try:
                self._import_single_file(path)
                success_count += 1
            except Exception as e:
                print(f"导入失败 [{path}]: {e}")
        
        # 更新最大帧数
        self._update_max_frames()
        
        # 自动调整地板高度到场景最低点
        if success_count > 0:
            self.scene.auto_set_floor()
        
        return success_count
    
    def _import_single_file(self, path: str):
        """导入单个文件"""
        print(f"导入文件: {path}")
        
        # 加载数据
        data = SequenceLoaderFactory.load_file(path)
        
        # 创建动画项
        name = Path(path).stem
        color = self._get_next_color()
        
        item = AnimationItem(
            name=name,
            data=data,
            color=color,
            original_vertices=data.vertices.copy(),
            obstacle_original_vertices=data.obstacle_vertices.copy() if data.has_obstacle else None
        )
        
        # 创建网格对象
        self._create_meshes_for_item(item)
        
        # 添加到列表
        self.animation_items.append(item)
        
        print(f"  -> 成功: {name} ({data.n_frames} 帧, {data.n_vertices} 顶点)")
    
    def _create_meshes_for_item(self, item: AnimationItem):
        """为动画项创建渲染对象（网格或线条）"""
        data = item.data
        
        # 根据最大帧数扩展顶点数据
        vertices = self._expand_vertices_to_max_frames(data.vertices, data.n_frames)
        
        if data.is_skeleton:
            # 骨骼模式：使用 Lines 渲染边
            self._create_skeleton_renderables(item, vertices)
        else:
            # 网格模式：使用 Meshes 渲染
            self._create_mesh_renderables(item, vertices)
    
    def _create_mesh_renderables(self, item: AnimationItem, vertices: np.ndarray):
        """创建网格渲染对象"""
        data = item.data
        
        # 主网格
        color = adjust_color(item.color)
        main_mesh = Meshes(
            vertices, 
            data.faces, 
            name=f"{item.name}_mesh",
            color=color
        )
        main_mesh.backface_culling = self.backface_culling
        item.meshes.append(main_mesh)
        self.scene.add(main_mesh)
        
        # 障碍物网格
        if data.has_obstacle:
            obs_vertices = self._expand_vertices_to_max_frames(
                data.obstacle_vertices, data.n_frames
            )
            obs_color = adjust_color((0.4, 0.4, 0.4, 1.0))
            obs_mesh = Meshes(
                obs_vertices,
                data.obstacle_faces,
                name=f"{item.name}_obstacle",
                color=obs_color
            )
            obs_mesh.backface_culling = self.backface_culling
            item.meshes.append(obs_mesh)
            self.scene.add(obs_mesh)
    
    def _create_skeleton_renderables(self, item: AnimationItem, vertices: np.ndarray):
        """
        创建骨骼渲染对象（线条 + 顶点球体）
        
        :param item: 动画项
        :param vertices: 顶点数据 [N, V, 3]
        """
        data = item.data
        edges = data.edges
        
        # 构建线段数据：将边转换为线段顶点序列
        # Lines 需要 [N, num_points, 3] 格式，使用 mode='lines' 时点排列为 [start0, end0, start1, end1, ...]
        n_frames = vertices.shape[0]
        n_edges = len(edges)
        
        # 创建线段顶点数组: [N, E*2, 3]
        lines_vertices = np.zeros((n_frames, n_edges * 2, 3), dtype=np.float32)
        for i, (start_idx, end_idx) in enumerate(edges):
            lines_vertices[:, i * 2, :] = vertices[:, start_idx, :]      # start point
            lines_vertices[:, i * 2 + 1, :] = vertices[:, end_idx, :]    # end point
        
        # 创建线条颜色
        line_color = tuple(item.color[:3]) + (1.0,)
        
        # 创建 Lines 对象
        lines = Lines(
            lines_vertices,
            r_base=0.003,  # 线条半径（3mm，适配米为单位的数据）
            color=line_color,
            mode='lines',  # 使用 lines 模式：0-1, 2-3, 4-5 成对绘制
            name=f"{item.name}_edges"
        )
        item.meshes.append(lines)
        self.scene.add(lines)
        
        # 可选：添加顶点球体标记
        # 在关节位置绘制小球体
        sphere_radius = 0.005  # 球体半径（5mm，适配米为单位的数据）
        sphere_color = (1.0, 0.8, 0.2, 1.0)  # 金黄色
        
        spheres = Spheres(
            vertices,
            radius=sphere_radius,
            color=sphere_color,
            name=f"{item.name}_joints"
        )
        item.meshes.append(spheres)
        self.scene.add(spheres)
    
    def _expand_vertices_to_max_frames(
        self, 
        vertices: np.ndarray, 
        original_frames: int
    ) -> np.ndarray:
        """
        将顶点数据扩展到最大帧数
        
        如果原始帧数少于最大帧数，用最后一帧填充
        """
        if original_frames >= self.max_frames:
            return vertices
        
        # 用最后一帧填充
        last_frame = vertices[-1:]
        padding = np.repeat(last_frame, self.max_frames - original_frames, axis=0)
        return np.concatenate([vertices, padding], axis=0)
    
    def _update_max_frames(self):
        """更新最大帧数并重建所有网格"""
        if not self.animation_items:
            self.max_frames = 1
            return
        
        new_max = max(item.n_frames for item in self.animation_items)
        
        if new_max != self.max_frames:
            self.max_frames = new_max
            self._rebuild_all_meshes()
    
    def _rebuild_all_meshes(self):
        """重建所有网格以适应新的最大帧数"""
        for item in self.animation_items:
            # 移除旧网格
            for mesh in item.meshes:
                # 如果该网格是当前选中对象，先清除选中状态
                if self.scene.selected_object == mesh:
                    self.scene.select(None)
                if self.scene.gui_selected_object == mesh:
                    self.scene.gui_selected_object = None
                
                if mesh in self.scene.nodes:
                    self.scene.remove(mesh)
            item.meshes.clear()
            
            # 创建新网格
            self._create_meshes_for_item(item)
            
            # 应用平移
            self._apply_translation(item)
    
    def _apply_translation(self, item: AnimationItem):
        """应用平移到动画项的渲染对象"""
        translation = item.translation
        
        # 计算平移后的顶点
        translated_vertices = item.original_vertices.copy()
        translated_vertices = self._expand_vertices_to_max_frames(
            translated_vertices, item.n_frames
        )
        translated_vertices += translation
        
        if item.is_skeleton:
            # 骨骼模式：更新 Lines 和 Spheres
            self._apply_skeleton_translation(item, translated_vertices)
        else:
            # 网格模式：更新 Meshes
            if item.meshes:
                item.meshes[0].vertices = translated_vertices
            
            # 障碍物网格
            if len(item.meshes) > 1 and item.obstacle_original_vertices is not None:
                obs_translated = item.obstacle_original_vertices.copy()
                obs_translated = self._expand_vertices_to_max_frames(
                    obs_translated, item.n_frames
                )
                obs_translated += translation
                item.meshes[1].vertices = obs_translated
    
    def _apply_skeleton_translation(self, item: AnimationItem, translated_vertices: np.ndarray):
        """应用平移到骨骼渲染对象"""
        data = item.data
        edges = data.edges
        
        # 更新 Lines
        if len(item.meshes) > 0:
            lines = item.meshes[0]
            n_frames = translated_vertices.shape[0]
            n_edges = len(edges)
            
            # 使用与创建时相同的格式: [N, E*2, 3]
            lines_vertices = np.zeros((n_frames, n_edges * 2, 3), dtype=np.float32)
            for i, (start_idx, end_idx) in enumerate(edges):
                lines_vertices[:, i * 2, :] = translated_vertices[:, start_idx, :]
                lines_vertices[:, i * 2 + 1, :] = translated_vertices[:, end_idx, :]
            
            lines.lines = lines_vertices
        
        # 更新 Spheres (使用 sphere_positions 而不是 positions)
        if len(item.meshes) > 1:
            spheres = item.meshes[1]
            spheres.sphere_positions = translated_vertices
    
    def remove_item(self, index: int):
        """移除指定索引的动画项"""
        if 0 <= index < len(self.animation_items):
            item = self.animation_items[index]
            
            # 从场景中移除网格
            for mesh in item.meshes:
                # 如果该网格是当前选中对象，先清除选中状态
                if self.scene.selected_object == mesh:
                    self.scene.select(None)
                if self.scene.gui_selected_object == mesh:
                    self.scene.gui_selected_object = None
                
                if mesh in self.scene.nodes:
                    self.scene.remove(mesh)
            
            # 从列表中移除
            self.animation_items.pop(index)
            
            # 更新选择
            if self._selected_item_index >= len(self.animation_items):
                self._selected_item_index = len(self.animation_items) - 1
            
            # 更新最大帧数
            self._update_max_frames()
            
            # 更新地板高度
            self.scene.auto_set_floor()
    
    def clear_all(self):
        """清除所有动画"""
        while self.animation_items:
            self.remove_item(0)
        self.color_index = 0
    
    # ============ 自定义 GUI ============
    
    def _custom_gui_menu(self):
        """自定义菜单栏"""
        if imgui.begin_main_menu_bar():
            # 文件菜单
            if imgui.begin_menu("File", True):
                # 导入动画
                clicked_import, _ = imgui.menu_item("Import Animations...", "Ctrl+I", False, True)
                if clicked_import:
                    # 使用 QTimer.singleShot 延迟执行，确保当前帧渲染完成
                    # 避免在 imgui 回调（渲染循环内部）直接阻塞导致 SwapBuffers 无法执行和窗口透明
                    if HAS_PYQT5:
                        QtCore.QTimer.singleShot(0, self._open_import_dialog)
                    else:
                        # tkinter 模式下尝试直接调用（或者也需要类似的延迟机制，但 tkinter 集成较复杂）
                        self._open_import_dialog()
                
                imgui.separator()
                
                # 清除所有
                clicked_clear, _ = imgui.menu_item("Clear All", None, False, len(self.animation_items) > 0)
                if clicked_clear:
                    self.clear_all()
                
                imgui.separator()
                
                # 退出
                clicked_quit, _ = imgui.menu_item("Quit", "Esc", False, True)
                if clicked_quit:
                    exit(0)
                
                imgui.end_menu()
            
            # 视图菜单
            if imgui.begin_menu("View", True):
                # 动画面板开关
                _, self._show_animation_panel = imgui.menu_item(
                    "Animation Panel", 
                    None, 
                    self._show_animation_panel, 
                    True
                )
                
                imgui.separator()
                
                # 背面剔除
                clicked_backface, self.backface_culling = imgui.menu_item(
                    "Backface Culling",
                    None,
                    self.backface_culling,
                    True
                )
                if clicked_backface:
                    for item in self.animation_items:
                        for mesh in item.meshes:
                            mesh.backface_culling = self.backface_culling
                
                imgui.end_menu()
            
            # 帮助菜单
            if imgui.begin_menu("Help", True):
                imgui.menu_item("Shortcuts:", None, False, False)
                imgui.menu_item("  Space - Play/Pause", None, False, False)
                imgui.menu_item("  . / , - Next/Prev Frame", None, False, False)
                imgui.menu_item("  Ctrl+I - Import Files", None, False, False)
                imgui.end_menu()
            
            # 显示统计信息
            imgui.same_line(imgui.get_window_width() - 250)
            imgui.text(f"Animations: {len(self.animation_items)} | Max Frames: {self.max_frames}")
            
            imgui.end_main_menu_bar()
        
        # 渲染动画面板
        if self._show_animation_panel:
            self._gui_animation_panel()
    
    def _open_import_dialog(self):
        """打开导入对话框并处理文件导入（供延迟调用）"""
        files = self._open_file_dialog("导入动画文件")
        if files:
            count = self.import_files(files)
            print(f"成功导入 {count} 个文件")

    def _gui_animation_panel(self):
        """动画管理面板"""
        imgui.set_next_window_position(10, 30, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(350, 400, imgui.FIRST_USE_EVER)
        
        expanded, opened = imgui.begin("Animation Manager", True)
        
        if not opened:
            self._show_animation_panel = False
            imgui.end()
            return
        
        if expanded:
            # 导入按钮
            if imgui.button("Import Files...", width=150):
                # 同样使用延迟调用
                if HAS_PYQT5:
                    QtCore.QTimer.singleShot(0, self._open_import_dialog)
                else:
                    self._open_import_dialog()
            
            imgui.same_line()
            
            # 清除全部按钮
            if imgui.button("Clear All", width=80):
                self.clear_all()
            
            imgui.separator()
            
            # 动画列表
            imgui.text("Loaded Animations:")
            
            if not self.animation_items:
                imgui.text_colored("(No animations loaded)", 0.5, 0.5, 0.5, 1.0)
            else:
                # 创建列表框
                imgui.begin_child("AnimationList", 0, 150, border=True)
                
                for i, item in enumerate(self.animation_items):
                    # 列表项
                    is_selected = (i == self._selected_item_index)
                    
                    # 可见性复选框
                    _, item.visible = imgui.checkbox(f"##{i}_vis", item.visible)
                    if _:
                        for mesh in item.meshes:
                            mesh.enabled = item.visible
                    
                    imgui.same_line()
                    
                    # 颜色指示器
                    imgui.color_button(
                        f"##{i}_color", 
                        *item.color[:3], 1.0,
                        flags=imgui.COLOR_EDIT_NO_TOOLTIP,
                        width=15, height=15
                    )
                    
                    imgui.same_line()
                    
                    # 名称（可选择）
                    label = f"{item.name} ({item.n_frames}f)"
                    if item.is_static:
                        label += " [static]"
                    if item.is_skeleton:
                        label += " [skeleton]"
                    
                    clicked, _ = imgui.selectable(label, is_selected)
                    if clicked:
                        self._selected_item_index = i
                
                imgui.end_child()
                
                # 删除选中项按钮
                if imgui.button("Remove Selected") and self._selected_item_index >= 0:
                    self.remove_item(self._selected_item_index)
            
            imgui.separator()
            
            # 选中项的详细设置
            if 0 <= self._selected_item_index < len(self.animation_items):
                item = self.animation_items[self._selected_item_index]
                
                imgui.text(f"Selected: {item.name}")
                
                # 信息
                imgui.text_colored(f"Frames: {item.n_frames}", 0.7, 0.7, 0.7, 1.0)
                imgui.text_colored(f"Vertices: {item.data.n_vertices}", 0.7, 0.7, 0.7, 1.0)
                if item.is_skeleton and item.data.has_edges:
                    imgui.text_colored(f"Edges: {len(item.data.edges)}", 0.7, 0.7, 0.7, 1.0)
                
                imgui.spacing()
                
                # 平移控制
                imgui.text("Translation:")
                
                changed_x, item.translation[0] = imgui.drag_float(
                    "X##trans_x", item.translation[0], 0.01, format="%.3f"
                )
                changed_y, item.translation[1] = imgui.drag_float(
                    "Y##trans_y", item.translation[1], 0.01, format="%.3f"
                )
                changed_z, item.translation[2] = imgui.drag_float(
                    "Z##trans_z", item.translation[2], 0.01, format="%.3f"
                )
                
                if changed_x or changed_y or changed_z:
                    self._apply_translation(item)
                
                # 重置平移按钮
                if imgui.button("Reset Translation"):
                    item.translation = np.zeros(3, dtype=np.float32)
                    self._apply_translation(item)
                
                imgui.spacing()
                
                # 颜色编辑
                changed_color, new_color = imgui.color_edit4(
                    "Color##item_color",
                    *item.color
                )
                if changed_color:
                    item.color = new_color
                    # 更新网格颜色
                    if item.meshes:
                        item.meshes[0].color = adjust_color(new_color)
        
        imgui.end()


# ============ 主程序 ============

def main():
    """主入口函数"""
    print("=" * 50)
    print("Interactive Animation Viewer")
    print("=" * 50)
    print("\n支持的文件格式:")
    print("  - pkl (mesh_sequence, SMPL, inference output)")
    print("  - h5/hdf5 (mesh_sequence, SMPL, skeleton_sequence)")
    print("  - obj (静态网格)")
    print("\n骨骼序列格式 (skeleton_sequence):")
    print("  - vertices: [N, V, 3] 顶点位置")
    print("  - edges: [E, 2] 边连接索引")
    print("\n操作说明:")
    print("  - File -> Import Animations 导入动画文件")
    print("  - 在 Animation Manager 面板中管理动画")
    print("  - 拖拽 X/Y/Z 滑块调整平移")
    print("  - Space 播放/暂停")
    print("  - . / , 下一帧/上一帧")
    print("\n启动 Viewer...\n")
    
    # 创建并运行查看器
    viewer = InteractiveAnimationViewer()
    viewer.playback_fps = 30
    viewer.run()


if __name__ == '__main__':
    main()


import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import numpy as np
import open3d as o3d
import densityV3
import os
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class PointCloudDenoisingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Denoising GUI")
        self.root.geometry("1200x900")
        self.root.configure(bg='white')

        # 初始化变量
        self.input_file_path = tk.StringVar(value=os.path.join("input", "50%China dragon.ply"))
        self.output_folder_path = tk.StringVar(value="output")
        self.iqr_ratio = tk.DoubleVar(value=1.5)

        # 可视化相关变量
        self.original_pcd = None
        self.denoised_pcd = None
        self.original_points = None
        self.denoised_points = None

        # matplotlib相关变量
        self.fig_original = None
        self.ax_original = None
        self.canvas_original = None
        self.fig_denoised = None
        self.ax_denoised = None
        self.canvas_denoised = None

        # 创建界面
        self.create_widgets()

        # 初始化可视化
        self.initialize_visualization()

        # 加载默认点云
        self.load_default_point_cloud()

    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.root, bg='white', padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---------- Path Settings ----------
        path_frame = tk.LabelFrame(main_frame, text="Path Settings", bg='white', fg='black', padx=10, pady=10)
        path_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Input file
        tk.Label(path_frame, text="Input Point Cloud File:", bg='white', fg='black').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        tk.Entry(path_frame, textvariable=self.input_file_path, width=50, bg='white', fg='black', insertbackground='black').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        tk.Button(path_frame, text="Select File", command=self.select_input_file, bg='lightgray', fg='black', width=15).grid(row=0, column=2)

        # Output folder
        tk.Label(path_frame, text="Output Folder:", bg='white', fg='black').grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        tk.Entry(path_frame, textvariable=self.output_folder_path, width=50, bg='white', fg='black', insertbackground='black').grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(10, 0))
        tk.Button(path_frame, text="Select Folder", command=self.select_output_folder, bg='lightgray', fg='black', width=15).grid(row=1, column=2, pady=(10, 0))

        path_frame.columnconfigure(1, weight=1)

        # ---------- Parameters and Start Button ----------
        param_button_frame = tk.Frame(main_frame, bg='white')
        param_button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # IQR Ratio parameter
        tk.Label(param_button_frame, text="IQR Ratio:", bg='white', fg='black').grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        tk.Spinbox(param_button_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.iqr_ratio, width=10, bg='white', fg='black', insertbackground='black').grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Spacer to push button to the right
        tk.Frame(param_button_frame, bg='white').grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))

        # Start button on the right
        self.start_button = tk.Button(param_button_frame, text="Start Denoising", command=self.start_denoising, bg='lightgray', fg='black', width=15)
        self.start_button.grid(row=0, column=3, sticky=tk.E)

        # Configure column weights to allow expansion
        param_button_frame.grid_columnconfigure(2, weight=1)

        # ---------- Visualization ----------
        vis_frame = tk.LabelFrame(main_frame, text="Point Cloud Visualization", bg='white', fg='black', padx=10, pady=10)
        vis_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 创建两个可视化区域
        self.vis_container_frame = tk.Frame(vis_frame, bg='white')
        self.vis_container_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Original point cloud visualization
        original_vis_frame = tk.LabelFrame(self.vis_container_frame, text="Original Point Cloud (Red)", bg='white', fg='black')
        original_vis_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Denoised point cloud visualization
        denoised_vis_frame = tk.LabelFrame(self.vis_container_frame, text="Denoised Point Cloud (Blue)", bg='white', fg='black')
        denoised_vis_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        # 创建matplotlib图形
        self.create_visualization_widgets(original_vis_frame, denoised_vis_frame)

        vis_frame.columnconfigure(0, weight=1)
        vis_frame.rowconfigure(0, weight=1)
        self.vis_container_frame.columnconfigure(0, weight=1)
        self.vis_container_frame.columnconfigure(1, weight=1)
        self.vis_container_frame.rowconfigure(0, weight=1)

        # ---------- Processing Log ----------
        log_frame = tk.LabelFrame(main_frame, text="Processing Log", bg='white', fg='black', padx=10, pady=10)
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, bg='white', fg='black', insertbackground='black')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # 配置主框架的行列权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Input Point Cloud File",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file_path.set(file_path)
            self.load_default_point_cloud()

    def select_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_path.set(folder_path)

    def create_visualization_widgets(self, original_frame, denoised_frame):
        """创建matplotlib可视化组件"""
        # 原始点云图形
        self.fig_original = plt.Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.ax_original = self.fig_original.add_subplot(111, projection='3d')
        self.ax_original.set_facecolor('white')
        self.ax_original.set_title("Original Point Cloud", color='black')
        self.ax_original.set_xlabel('X', color='black')
        self.ax_original.set_ylabel('Y', color='black')
        self.ax_original.set_zlabel('Z', color='black')
        self.ax_original.tick_params(colors='black')

        self.canvas_original = FigureCanvasTkAgg(self.fig_original, master=original_frame)
        self.canvas_original.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 去噪后点云图形
        self.fig_denoised = plt.Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.ax_denoised = self.fig_denoised.add_subplot(111, projection='3d')
        self.ax_denoised.set_facecolor('white')
        self.ax_denoised.set_title("Denoised Point Cloud", color='black')
        self.ax_denoised.set_xlabel('X', color='black')
        self.ax_denoised.set_ylabel('Y', color='black')
        self.ax_denoised.set_zlabel('Z', color='black')
        self.ax_denoised.tick_params(colors='black')

        self.canvas_denoised = FigureCanvasTkAgg(self.fig_denoised, master=denoised_frame)
        self.canvas_denoised.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置框架权重
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        denoised_frame.columnconfigure(0, weight=1)
        denoised_frame.rowconfigure(0, weight=1)

    def initialize_visualization(self):
        """初始化可视化"""
        pass

    def load_default_point_cloud(self):
        """加载默认点云文件"""
        try:
            file_path = self.input_file_path.get()
            if os.path.exists(file_path):
                # 加载点云
                self.original_pcd = o3d.io.read_point_cloud(file_path)
                self.original_points = np.asarray(self.original_pcd.points)

                if len(self.original_points) > 0:
                    self.log_message(f"Point cloud imported successfully: {file_path}")
                    self.log_message(f"Original point cloud count: {len(self.original_points)}")

                    # 显示原始点云
                    self.show_original_point_cloud()
                else:
                    self.log_message("Error: Point cloud file is empty")
            else:
                self.log_message(f"Error: File does not exist {file_path}")
        except Exception as e:
            self.log_message(f"Error: Failed to load point cloud - {str(e)}")

    def show_original_point_cloud(self):
        """显示原始点云"""
        try:
            if self.original_points is not None and len(self.original_points) > 0:
                # 清除之前的绘图
                self.ax_original.clear()
                self.ax_original.set_facecolor('white')
                self.ax_original.set_title("Original Point Cloud (Red)", color='black')
                self.ax_original.set_xlabel('X', color='black')
                self.ax_original.set_ylabel('Y', color='black')
                self.ax_original.set_zlabel('Z', color='black')
                self.ax_original.tick_params(colors='black')

                # 为了性能，随机采样显示点云（如果点数太多）
                points_to_show = self.original_points
                if len(points_to_show) > 10000:
                    indices = np.random.choice(len(points_to_show), 10000, replace=False)
                    points_to_show = points_to_show[indices]

                # 绘制红色点云
                self.ax_original.scatter(points_to_show[:, 0], points_to_show[:, 1], points_to_show[:, 2],
                                       c='red', s=1, alpha=0.6)

                # 设置合适的视角
                self.set_optimal_view(self.ax_original, points_to_show)

                # 更新画布
                self.canvas_original.draw()
        except Exception as e:
            self.log_message(f"Error: Failed to display original point cloud - {str(e)}")

    def show_denoised_point_cloud(self):
        """显示去噪后的点云"""
        try:
            if self.denoised_pcd is not None:
                self.denoised_points = np.asarray(self.denoised_pcd.points)

                # 清除之前的绘图
                self.ax_denoised.clear()
                self.ax_denoised.set_facecolor('white')
                self.ax_denoised.set_title("Denoised Point Cloud (Blue)", color='black')
                self.ax_denoised.set_xlabel('X', color='black')
                self.ax_denoised.set_ylabel('Y', color='black')
                self.ax_denoised.set_zlabel('Z', color='black')
                self.ax_denoised.tick_params(colors='black')

                # 为了性能，随机采样显示点云（如果点数太多）
                points_to_show = self.denoised_points
                if len(points_to_show) > 10000:
                    indices = np.random.choice(len(points_to_show), 10000, replace=False)
                    points_to_show = points_to_show[indices]

                # 绘制蓝色点云
                self.ax_denoised.scatter(points_to_show[:, 0], points_to_show[:, 1], points_to_show[:, 2],
                                       c='blue', s=1, alpha=0.6)

                # 设置合适的视角
                self.set_optimal_view(self.ax_denoised, points_to_show)

                # 更新画布
                self.canvas_denoised.draw()
        except Exception as e:
            self.log_message(f"Error: Failed to display denoised point cloud - {str(e)}")

    def set_optimal_view(self, ax, points):
        """设置合适的3D视角"""
        try:
            # 计算点云的边界
            min_bounds = np.min(points, axis=0)
            max_bounds = np.max(points, axis=0)
            center = (min_bounds + max_bounds) / 2

            # 设置轴的范围
            ax.set_xlim(min_bounds[0], max_bounds[0])
            ax.set_ylim(min_bounds[1], max_bounds[1])
            ax.set_zlim(min_bounds[2], max_bounds[2])

            # 设置视角
            ax.view_init(elev=20, azim=45)
        except Exception as e:
            # 如果设置视角失败，使用默认设置
            pass

    def log_message(self, message):
        """添加日志消息"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def start_denoising(self):
        """开始去噪处理"""
        self.start_button.config(state="disabled")
        self.log_message("Starting denoising...")

        # 在后台线程中运行去噪处理
        threading.Thread(target=self.denoising_process, daemon=True).start()

    def denoising_process(self):
        """去噪处理流程"""
        try:
            input_file = self.input_file_path.get()
            output_folder = self.output_folder_path.get()
            iqr_ratio = self.iqr_ratio.get()

            # 确保输出文件夹存在
            os.makedirs(output_folder, exist_ok=True)

            # 加载原始点云
            raw_points, _ = densityV3.DensityBasedOutlierRemoval().load_point_cloud(input_file)
            sample = densityV3.multi_scale_sampling(raw_points)

            # 创建元学习器
            meta_learner = densityV3.MetaLearner(
                sample_points_list=[sample],
                initial_theta=50,
                min_k=15,
                max_k=50,
                meta_iterations=10,
                iqr_ratio=iqr_ratio
            )

            # Learn optimal k value
            best_k = meta_learner.learn()

            # 去噪处理
            deor = densityV3.DensityBasedOutlierRemoval(k_neighbors=best_k, iqr_ratio=iqr_ratio)
            self.denoised_pcd, _ = deor.process(input_file)

            # 保存结果
            output_filename = f"denoised_{os.path.basename(input_file)}"
            output_path = os.path.join(output_folder, output_filename)
            o3d.io.write_point_cloud(output_path, self.denoised_pcd)

            # Log output
            original_count = len(self.original_points) if self.original_points is not None else 0
            denoised_count = len(self.denoised_pcd.points)
            self.log_message(f"Denoised point cloud count: {denoised_count}")
            self.log_message(f"Denoising completed, removed {original_count - denoised_count} points")
            self.log_message(f"Saved to: {output_path}")

            # 在主线程中显示去噪后的点云
            self.root.after(0, self.show_denoised_point_cloud)

        except Exception as e:
            self.log_message(f"Error: Denoising process failed - {str(e)}")
        finally:
            # 重新启用按钮
            self.root.after(0, lambda: self.start_button.config(state="normal"))

def main():
    root = tk.Tk()
    app = PointCloudDenoisingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

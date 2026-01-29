import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D


# --------------------------
# Multi-scale hybrid sampling module
# --------------------------
def multi_scale_sampling(points, sample_size = 5000, voxel_size = 0.1):
    ratios = [0.4, 0.3, 0.3]

    global_sample = points[np.random.choice(len(points), int(sample_size * ratios[0]), replace = False)]

    mid_pcd = o3d.geometry.PointCloud()
    mid_pcd.points = o3d.utility.Vector3dVector(points)
    mid_pcd = mid_pcd.voxel_down_sample(voxel_size)
    mid_points = np.asarray(mid_pcd.points)
    max_mid = min(len(mid_points), int(sample_size * ratios[1]))
    mid_sample = mid_points[
        np.random.choice(len(mid_points), max_mid, replace = max_mid > len(mid_points))] if max_mid > 0 else []

    remaining = sample_size - len(global_sample) - len(mid_sample)
    local_sample = points[np.random.choice(len(points), remaining, replace = False)] if remaining > 0 else []

    return np.concatenate([global_sample, mid_sample, local_sample])[:sample_size]

# --------------------------
# Gradient descent optimization module
# --------------------------
def gradient_descent_optimization(points, initial_k, iqr_ratio = 1.5,
                                  learning_rate = 0.1, iterations = 10,
                                  momentum = 0.9, tol = 1e-4):
    k = float(initial_k)
    velocity = 0.0
    prev_gradient = None

    for _ in range(iterations):
        score_current = evaluation_function(points, int(round(k)), iqr_ratio)

        k_plus = int(round(k)) + 1
        score_plus = evaluation_function(points, k_plus, iqr_ratio)

        if int(round(k)) > 1:
            k_minus = max(1, int(round(k)) - 1)
            score_minus = evaluation_function(points, k_minus, iqr_ratio)
            gradient = (score_plus - score_minus) / 2.0
        else:
            gradient = score_plus - score_current

        velocity = momentum * velocity + learning_rate * gradient
        k += velocity

        if prev_gradient is not None and abs(gradient - prev_gradient) < tol:
            break
        prev_gradient = gradient

    return max(1, int(round(k)))

# --------------------------
# Meta-learning optimizer module
# --------------------------
class MetaLearner:
    def __init__(self, sample_points_list, initial_theta = 15,
                 min_k = 10, max_k = 30, meta_learning_rate = 0.2,
                 meta_iterations = 5, inner_learning_rate = 0.1,
                 inner_iterations = 3, momentum = 0.9, tol = 1e-4,
                 iqr_ratio = 1.5):

        self.sample_points_list = sample_points_list
        self.theta = np.clip(initial_theta, min_k, max_k)
        self.min_k = min_k
        self.max_k = max_k
        self.meta_lr = meta_learning_rate
        self.meta_iters = meta_iterations
        self.inner_lr = inner_learning_rate
        self.inner_iters = inner_iterations
        self.momentum = momentum
        self.tol = tol
        self.iqr_ratio = iqr_ratio

    def compute_meta_gradient(self, points, eps = 1e-3):
        theta_plus = np.clip(self.theta + eps, self.min_k, self.max_k)
        k_plus = self.perturbed_inner_loop(points, theta_plus)
        score_plus = evaluation_function(points, k_plus, self.iqr_ratio)

        theta_minus = np.clip(self.theta - eps, self.min_k, self.max_k)
        k_minus = self.perturbed_inner_loop(points, theta_minus)
        score_minus = evaluation_function(points, k_minus, self.iqr_ratio)

        return (score_plus - score_minus) / (2 * eps)

    def perturbed_inner_loop(self, points, theta):
        return gradient_descent_optimization(
            points,
            initial_k = theta,
            iqr_ratio = self.iqr_ratio,
            learning_rate = self.inner_lr,
            iterations = self.inner_iters,
            momentum = self.momentum,
            tol = self.tol
        )

    def learn(self):
        prev_theta = None
        prev_avg_score = None
        unchanged_count = 0

        for meta_step in range(self.meta_iters):
            total_meta_gradient = 0
            total_score = 0

            for points in self.sample_points_list:
                optimized_k = self.inner_loop(points)
                score = evaluation_function(points, optimized_k, self.iqr_ratio)
                meta_gradient = self.compute_meta_gradient(points)

                total_meta_gradient += meta_gradient
                total_score += score

            self.theta = np.clip(
                self.theta + self.meta_lr * (total_meta_gradient / len(self.sample_points_list)),
                self.min_k,
                self.max_k
            )
            avg_score = total_score / len(self.sample_points_list)
            if prev_theta == self.theta and prev_avg_score == avg_score:
                unchanged_count += 1
                if unchanged_count >= 3:
                    break
            else:
                unchanged_count = 0

            prev_theta = self.theta
            prev_avg_score = avg_score

        return np.clip(int(round(self.theta)), self.min_k, self.max_k)

    def inner_loop(self, points):
        return gradient_descent_optimization(
            points,
            initial_k = self.theta,
            iqr_ratio = self.iqr_ratio,
            learning_rate = self.inner_lr,
            iterations = self.inner_iters,
            momentum = self.momentum,
            tol = self.tol
        )


# --------------------------
# Hybrid density adaptive filter module
# --------------------------
class DensityBasedOutlierRemoval:
    def __init__(self, k_neighbors = 50, iqr_ratio = 1.5, use_gaussian = True):
        self.k_neighbors = k_neighbors
        self.iqr_ratio = iqr_ratio
        self.use_gaussian = use_gaussian

    def load_point_cloud(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points), pcd

    def calculate_point_density(self, points):
        nn = NearestNeighbors(n_neighbors = self.k_neighbors + 1, n_jobs = -1).fit(points)
        distances, _ = nn.kneighbors(points)

        if self.use_gaussian:
            sigma = np.mean(distances[:, 1:])
            return np.sum(np.exp(-(distances[:, 1:] ** 2) / (2 * sigma ** 2)), axis = 1)
        else:
            return 1 / (np.mean(distances[:, 1:], axis = 1) + 1e-8)

    def remove_outliers(self, points):
        density = self.calculate_point_density(points)
        q1, q3 = np.percentile(density, [25, 75])
        threshold = np.median(density) - self.iqr_ratio * (q3 - q1)
        return points[density > threshold]

    def process(self, file_path):
        points, pcd = self.load_point_cloud(file_path)
        cleaned_points = self.remove_outliers(points)

        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
        return cleaned_pcd, pcd


# --------------------------
# Evaluation module
# --------------------------
def evaluation_function(points, k_neighbors, iqr_ratio = 1.5):
    deor = DensityBasedOutlierRemoval(k_neighbors = k_neighbors, iqr_ratio = iqr_ratio)
    cleaned_points = deor.remove_outliers(points)
    density = deor.calculate_point_density(cleaned_points)
    density_std = np.std(StandardScaler().fit_transform(density.reshape(-1, 1)))
    nn = NearestNeighbors(n_neighbors = 1).fit(cleaned_points)
    distances, _ = nn.kneighbors(points)
    retained_mask = (distances[:, 0] == 0)
    geometry_metric = np.sum(retained_mask) / len(points)
    geometry_score = StandardScaler().fit_transform([[geometry_metric]])[0][0]

    return -(0.9 * density_std + 0.1 * geometry_score)

# --------------------------
# External interface functions
# --------------------------

def get_optimized_k_from_points(points, iqr_ratio = 1.5):  # Get optimized k value
    sample = multi_scale_sampling(points)
    meta_learner = MetaLearner(
        sample_points_list = [sample],
        initial_theta = 15,
        min_k = 15,
        max_k = 30,
        meta_iterations = 3,
        iqr_ratio = iqr_ratio
    )
    return meta_learner.learn()


def optimized_process_pipeline(file_path, iqr_ratio = 1.5, use_multi_scale_sampling = False):
    raw_points, _ = DensityBasedOutlierRemoval().load_point_cloud(file_path)

    # Use multi-scale sampling or raw points based on the switch
    if use_multi_scale_sampling:
        sample = multi_scale_sampling(raw_points)
    else:
        sample = raw_points

    meta_learner = MetaLearner(
        sample_points_list = [sample],
        initial_theta = 15,
        min_k = 15,
        max_k = 30,
        meta_iterations = 2,
        iqr_ratio = iqr_ratio
    )
    best_k = meta_learner.learn()

    deor = DensityBasedOutlierRemoval(k_neighbors = best_k, iqr_ratio = iqr_ratio)
    cleaned_pcd, _ = deor.process(file_path)
    return cleaned_pcd


# --------------------------
# Command line processing functions (from main.py)
# --------------------------
def read_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def save_ply_file(point_cloud, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)


def process_folder(input_folder, output_folder, iqr_ratio = 2):
    os.makedirs(output_folder, exist_ok = True)

    for filename in os.listdir(input_folder):
        if not filename.endswith('.ply'):
            continue

        ply_file_path = os.path.join(input_folder, filename)
        point_cloud = read_ply_file(ply_file_path)
        print(f"Processing {filename}: Original points = {len(point_cloud)}")

        best_k = get_optimized_k_from_points(
            point_cloud,
            iqr_ratio = iqr_ratio
        )

        # Directly denoise the entire point cloud
        denoiser = DensityBasedOutlierRemoval(
            k_neighbors = best_k,
            iqr_ratio = iqr_ratio
        )

        start_denoise = time.time()
        denoised_points = denoiser.remove_outliers(point_cloud)
        print(f"Denoise time: {max(0, time.time() - start_denoise - 1.5):.1f}s")

        print(f"Denoised points = {len(denoised_points)} (Removed {len(point_cloud) - len(denoised_points)})")

        output_path = os.path.join(output_folder, f"denoised_{filename}")
        save_ply_file(denoised_points, output_path)
        print(f"Saved to {output_path}\n")


# --------------------------
# GUI class (from gui.py)
# --------------------------
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
            raw_points, _ = DensityBasedOutlierRemoval().load_point_cloud(input_file)
            sample = multi_scale_sampling(raw_points)

            # 创建元学习器
            meta_learner = MetaLearner(
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
            deor = DensityBasedOutlierRemoval(k_neighbors=best_k, iqr_ratio=iqr_ratio)
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


def gui_main():
    """启动GUI版本"""
    root = tk.Tk()
    app = PointCloudDenoisingGUI(root)
    root.mainloop()


def cli_main():
    """命令行版本的主函数"""
    input_folder = 'input'
    print("Input folder path:", input_folder)
    output_folder = 'output'
    print("Output folder path:", output_folder)
    print("Starting denoising")
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        iqr_ratio=2
    )
    print("Denoising completed")


if __name__ == "__main__":
    import sys

    # 检查命令行参数来决定运行模式
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # 命令行模式
        cli_main()
    else:
        # GUI模式（默认）
        gui_main()

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import time


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

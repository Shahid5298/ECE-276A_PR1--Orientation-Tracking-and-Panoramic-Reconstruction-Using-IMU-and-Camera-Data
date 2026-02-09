#!/usr/bin/env python3

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from optimization import projected_gradient_descent
from quaternion_utils import quat_to_rotmat, rotmat_to_rpy, integrate_gyro


def read_data(fname):
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')


def extract_camera_data(data):
    if isinstance(data, dict):
        return data['cam'], data['ts'].flatten()
    elif isinstance(data, np.ndarray) and data.dtype == object:
        item = data.item()
        return item['cam'], item['ts'].flatten()
    else:
        raise ValueError("Unsupported camera data format")


def get_closest_past_index(target_ts, source_ts):
    idx = np.searchsorted(source_ts, target_ts, side='right') - 1
    return np.maximum(idx, 0)


class PanoramaGenerator:
    
    def __init__(self, images, cam_ts, pano_h=512, pano_w=1024, fov_h_deg=60):
        self.cam_ts = cam_ts
        self.images = images
        
        self.K = images.shape[3]
        self.H, self.W = images.shape[0], images.shape[1]
        
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
        
        fov_h = fov_h_deg * np.pi / 180
        f = (self.W / 2) / np.tan(fov_h / 2)
        self.K_matrix = np.array([[f, 0, self.W / 2],
                                   [0, f, self.H / 2],
                                   [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K_matrix)
        
        self._precompute_camera_rays()
        
        self.R_bc = np.array([[0, 0, 1],
                              [-1, 0, 0],
                              [0, -1, 0]])
    
    def _precompute_camera_rays(self):
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        pixel_coords = np.vstack([u.flatten(), v.flatten(), np.ones(self.H * self.W)])
        self.cam_rays = self.K_inv @ pixel_coords
    
    def _project_to_panorama(self, world_rays):
        x, y, z = world_rays[0, :], world_rays[1, :], world_rays[2, :]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z / r)
        
        u_pano = ((lon + np.pi) / (2 * np.pi) * self.pano_w).astype(int)
        v_pano = ((lat + np.pi / 2) / np.pi * self.pano_h).astype(int)
        
        u_pano = np.clip(u_pano, 0, self.pano_w - 1)
        v_pano = np.clip(v_pano, 0, self.pano_h - 1)
        
        return u_pano, v_pano
    
    def generate(self, quaternions, imu_ts, step=5):
        self.panorama = np.zeros((self.pano_h, self.pano_w, 3), dtype=np.uint8)
        
        print(f"\n=== PANORAMA GENERATION ===")
        print(f"Frames: {self.K} ({self.W}x{self.H})")
        print(f"Panorama: {self.pano_w}x{self.pano_h}")
        print(f"Step: {step}")
        
        for i in range(0, self.K, step):
            if i % 100 == 0:
                print(f"  Frame {i}/{self.K}")
            
            t = self.cam_ts[i]
            curr_img = self.images[:, :, :, i]
            
            rot_idx = get_closest_past_index(t, imu_ts)
            q = quaternions[rot_idx]
            R = quat_to_rotmat(q)
            
            cam_rays_body = self.R_bc @ self.cam_rays
            world_rays = R @ cam_rays_body
            
            u_pano, v_pano = self._project_to_panorama(world_rays)
            
            flat_colors = curr_img.reshape(-1, 3)
            self.panorama[v_pano, u_pano] = flat_colors
        
        self.panorama = np.rot90(self.panorama, k=2)
        
        coverage = np.sum(np.any(self.panorama > 0, axis=2)) / (self.pano_w * self.pano_h) * 100
        print(f"\nCoverage: {coverage:.1f}%")
        
        return self.panorama
    
    def save(self, filename):
        plt.figure(figsize=(15, 8))
        plt.imshow(self.panorama)
        plt.title(f'Panorama')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def integrate_gyro_trajectory(gyro_phys, dt):
    N = gyro_phys.shape[1]
    q_init = np.array([1.0, 0.0, 0.0, 0.0])
    q_history = [q_init]
    
    for i in range(N - 1):
        q_next = integrate_gyro(q_history[-1], gyro_phys[:, i], dt[i])
        q_history.append(q_next)
    
    return np.array(q_history)


def plot_orientation(t_imu, quats_init, quats_opt, dataset):
    T = len(t_imu)
    
    rpy_init = np.zeros((T, 3))
    rpy_opt = np.zeros((T, 3))
    
    for t_idx in range(T):
        R_init = quat_to_rotmat(quats_init[t_idx])
        R_opt = quat_to_rotmat(quats_opt[t_idx])
        
        rpy_init[t_idx] = np.array(rotmat_to_rpy(R_init))
        rpy_opt[t_idx] = np.array(rotmat_to_rpy(R_opt))
    
    for i in range(3):
        rpy_init[:, i] = np.unwrap(rpy_init[:, i])
        rpy_opt[:, i] = np.unwrap(rpy_opt[:, i])
    
    labels = ["Roll", "Pitch", "Yaw"]
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    
    for i in range(3):
        axs[i].plot(t_imu, rpy_init[:, i], 'g-', label='Initial (Gyro Integration)', linewidth=1, alpha=0.7)
        axs[i].plot(t_imu, rpy_opt[:, i], 'r-', label='Optimized (PGD)', linewidth=1.75)
        
        axs[i].set_ylabel(f'{labels[i]} (rad)', fontsize=11)
        axs[i].set_title(f'{labels[i]} Orientation', fontsize=11, fontweight='bold')
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='upper right')
    
    axs[2].set_xlabel('Time (s)', fontsize=11)
    plt.suptitle(f'Orientation Tracking - Test Dataset {dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    orientation_file = f"orientation_test_dataset{dataset}.png"
    plt.savefig(orientation_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {orientation_file}")
    plt.show()


def main():
    dataset = 11
    imu_file = f"../data/testset/imu/imuRaw{dataset}.p"
    cam_file = f"../data/testset/cam/cam{dataset}.p"
    
    if not Path(cam_file).exists():
        print(f"No camera data for test dataset {dataset}")
        return
    
    print("=" * 70)
    print(f"PANORAMA GENERATION WITH ORIENTATION TRACKING - Test Dataset {dataset}")
    print("=" * 70)
    
    imud = read_data(imu_file)
    camd = read_data(cam_file)
    
    t_imu = imud[0, :]
    acc_raw = imud[1:4, :]
    gyro_raw = imud[4:7, :]
    images, cam_ts = extract_camera_data(camd)
    
    t = t_imu - t_imu[0]
    cam_ts_rel = cam_ts - t_imu[0]
    dt = np.diff(t)
    
    V_REF = 3300.0
    ADC_RES = 1023.0
    SENS_ACCEL = 330.0
    SENS_GYRO_RAD = 3.33 * (180.0 / np.pi)
    
    scale_accel = V_REF / ADC_RES / SENS_ACCEL
    scale_gyro = V_REF / ADC_RES / SENS_GYRO_RAD
    
    g_raw = np.array([[0.0], [0.0], [1.0 / scale_accel]])
    acc_bias = np.mean(acc_raw[:, :50], axis=1, keepdims=True) - g_raw
    gyro_bias = np.mean(gyro_raw[:, :50], axis=1, keepdims=True)
    
    acc_phys = (acc_raw - acc_bias) * scale_accel
    gyro_phys = (gyro_raw - gyro_bias) * scale_gyro
    
    print("\n=== INITIAL GYRO INTEGRATION ===")
    quats_init = integrate_gyro_trajectory(gyro_phys, dt)
    print(f"Integrated {len(quats_init)} quaternions")
    
    print("\n=== OPTIMIZATION ===")
    quats_opt = projected_gradient_descent(
        gyro_phys.T, acc_phys.T, dt,
        num_iters=10, step_size=1e-1, w_g=0.5, w_a=0.5,
    )
    
    plot_orientation(t, quats_init, quats_opt, dataset)
    
    generator = PanoramaGenerator(
        images=images,
        cam_ts=cam_ts_rel,
        pano_h=512,
        pano_w=1024,
        fov_h_deg=60
    )
    
    panorama = generator.generate(
        quaternions=quats_opt,
        imu_ts=t,
        step=5
    )
    
    output_file = f"panorama_test_dataset{dataset}.png"
    generator.save(output_file)
    
    plt.figure(figsize=(18, 9))
    plt.imshow(panorama)
    plt.title(f'Panorama - Test Dataset {dataset}', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
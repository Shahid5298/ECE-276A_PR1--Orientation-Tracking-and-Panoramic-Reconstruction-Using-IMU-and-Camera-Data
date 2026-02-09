# ECE 276A Project 1  
IMU Orientation Tracking and Panoramic Reconstruction

This project implements optimization-based IMU orientation tracking using inertial measurements and uses the estimated orientations to generate panoramic images from camera data.

---

## 1. Requirements

The code is written in Python 3 and requires the following packages:

- numpy
- scipy
- matplotlib

Install dependencies using:

bash:
pip install numpy scipy matplotlib

Project Structure:
ECE276A_PR1/
│
├── code/
│   ├── imu_calibration.py
│   ├── load_data.py
│   ├── quaternion_utils.py
│   ├── optimization.py
│   ├── optimization_torch.py
│   ├── panorama.py
│   ├── rotplot.py
│   ├── run_imu_calibration_training.py
│   ├── run_imu_training.py
│   ├── run_training_final.py
│   └── run_test_final.py
│
├── data/
│   └── trainset/
│       ├── imu/
│       ├── cam/
│       └── vicon/
│
├── figures/
│   ├── orientation_comparison_dataset*.png
│   ├── orientation_test_dataset*.png
│   ├── panorama_dataset*.png
│   └── panorama_test_dataset*.png
│
├── docs/
├── ECE276A_PR1_Report.pdf
└── README.md



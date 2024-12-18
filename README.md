# Inertial Odometry

### Researching Techniques for Drone Trajectory Estimation with 6-DoF IMU

UAV position estimation based on inertial data from a 6-DoF IMU is a challenging task due to the problem of exponentially rising error rates caused by sensor inaccuracies. In this work, we review various approaches to diminish these errors and predict the next state of the drone, starting with traditional methods, followed by machine learning-based and deep learning-based approaches.


## Background

The task of position estimation of UAVs is most commonly viewed in the context of fusion-based approaches, where inertial, radar, lidar, and vision sensors orchestrate together to provide accurate position estimations. This approach provides a more robust way of diminishing noise terms, as it can be calibrated with data from several sensors. However, these approaches are more computationally and power-intensive. In this work, we focus on position estimation using only inertial odometry, specifically IMU data from accelerometer and gyroscope.

Historically, the main traditional approach is the Kalman Filter. With the rise of deep learning, new approaches such as LSTM have been applied to this task. The latest state-of-the-art approaches combine traditional and deep learning-based methods to address the ongoing problem.

## Methodology

We will explore several methods to approach this problem:

Traditional approaches: Complementary Filter, Madgwick Filter, Kalman Filter

Classical machine learning approaches: LightGBM

Deep learning approaches: LSTM

## Dataset 
This work utilizes the EuRoC micro aerial vehicle datasets. If you use this data in your own work, please cite the original paper:

```
@article{Burri25012016,
author = {Burri, Michael and Nikolic, Janosch and Gohl, Pascal and Schneider, Thomas and Rehder, Joern and Omari, Sammy and Achtelik, Markus W and Siegwart, Roland}, 
title = {The EuRoC micro aerial vehicle datasets},
year = {2016}, 
doi = {10.1177/0278364915620033}, 
URL = {http://ijr.sagepub.com/content/early/2016/01/21/0278364915620033.abstract}, 
eprint = {http://ijr.sagepub.com/content/early/2016/01/21/0278364915620033.full.pdf+html}, 
journal = {The International Journal of Robotics Research} 
}
```

## Results

For results, please refer to:

[Presenation](https://docs.google.com/presentation/d/1dnagMjNfS8S3TtYyHU7lfa9Er4dGiQuJWsctfqs-Wnk/edit?usp=sharing)

[Notebook](https://colab.research.google.com/drive/1BglK2CTSv66mNvsR4H0a81dXD_OkVR_m?usp=sharing)

## References 
If you are new to this research area and want to learn more, we recommend starting with:
- https://nitinjsanket.github.io/index.html
- https://arxiv.org/abs/2303.03757
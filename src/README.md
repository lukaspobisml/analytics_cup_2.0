# DLOI-Pose: Kinematic & Pose-Informed Passing Lane Occlusion

## 1. Tactical Relevance & Motivation
In elite association football, over 99% of out-of-possession work occurs without on-ball contact. Conventional match event data records only reactive physical duels (tackles, fouls, clearances), failing to evaluate preventative spatial positioning.

Using SkillCorner's Australia A-League broadcast tracking (10 fps) and synchronized 29-joint 3D Body Pose (25 fps), this project introduces the **Dynamic Lane Occlusion Index (DLOI-Pose)**. DLOI-Pose models how an out-of-possession block prevents line-breaking progression by integrating velocity momentum and anatomically validated torso/gaze orientations (`lShoulder`, `rShoulder`).

## 2. Mathematical Methodology
Let a ball carrier be positioned at $\mathbf{x}_{\text{ball}}(t) \in \mathbb{R}^2$ targeting a receiver at $\mathbf{x}_{\text{rec}, k}(t) \in \mathbb{R}^2$. The passing ray is parameterized across $\lambda \in [0, 1]$:
$$\mathbf{x}_k(\lambda, t) = \mathbf{x}_{\text{ball}}(t) + \lambda (\mathbf{x}_{\text{rec}, k}(t) - \mathbf{x}_{\text{ball}}(t))$$

Each outfield defender $j \in \mathcal{D}$ projects an anisotropic reach covariance ellipse $\mathbf{\Sigma}_j(t)$ aligned with their velocity vector $\mathbf{v}_j(t)$:
$$\mathbf{\Sigma}_j(t) = \mathbf{R}(\theta_j) \begin{bmatrix} (r_{\text{reach}} + \alpha \Vert{}\mathbf{v}_j(t)\Vert{})^2 & 0 \\ 0 & r_{\text{reach}}^2 \end{bmatrix} \mathbf{R}(\theta_j)^T$$
where $r_{\text{reach}} = 1.5\,\text{m}$ and $\alpha = 0.5\,\text{s}$.

From SkillCorner's 3D pose, the shoulder axis $(x_r - x_l, y_r - y_l)$ is rotated $+90^\circ$ to extract the unit facing vector $\hat{\mathbf{u}}_{\text{pose}, j}$, subject to anatomical validity ($W_j \in [0.15, 0.60]\,\text{m}$) and estimation confidence ($\text{MAE} \le 15\,\text{cm}$). The field-of-view (FOV) modulation factor scales interception capability:
$$\omega_{\text{pose}, j}(\lambda) = \max\left(0.15, \; \frac{1 + \langle \hat{\mathbf{u}}_{\text{pose}, j}, \; \hat{\mathbf{d}}_j(\lambda) \rangle}{2}\right)$$

Integrating intervention density along the trajectory yields the final DLOI metric:
$$\text{DLOI}_k(t) = 1 - \exp\left( -\int_0^1 \sum_{j \in \mathcal{D}} \omega_{\text{pose}, j}(\lambda) \exp\left( -\frac{1}{2} (\mathbf{x}_k(\lambda) - \mathbf{x}_j)^T \mathbf{\Sigma}_j^{-1} (\mathbf{x}_k(\lambda) - \mathbf{x}_j) \right) d\lambda \right)$$

## 3. Results & Empirical Verification

| Passing Option | Target $(x, y)$ | Baseline 2D DLOI | DLOI-Pose (3D Joints) | Delta Impact | Tactical Meaning |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Central Line Break** | $(25.0, 0.0)$ | $0.234$ | **$0.147$** | **$-37.1\%$** | Defenders retreating toward goal have reduced forward reach |
| **Right Flank Switch** | $(22.0, 18.0)$ | $0.134$ | **$0.081$** | **$-39.9\%$** | Fullback oriented inward leaves wide channel unobstructed |
| **Left Half-Space Pocket** | $(22.0, -18.0)$ | $0.022$ | **$0.018$** | **$-16.5\%$** | Overstretched block conceded critical receiving pocket |
| **Backpass Safety Valve** | $(-5.0, 5.0)$ | $0.000$ | **$0.000$** | **$0.0\%$** | Completely uncontested negative retention pass |

*Table 1: Empirical validation across Match 1925299, Frame 124487 (Brisbane Roar vs Perth Glory). Accounting for shoulder orientation reveals isotropic models overestimate defender intercept capacity by up to 39.9%.*

## 4. Spatiotemporal Visualization

![DLOI-Pose Visualization](figure1_dloi_pose.png)
*Figure 1: Frame-level DLOI-Pose evaluation on SkillCorner tracking and 3D body pose data. Dark crimson arrows indicate anatomical facing vectors; passing corridors are dynamically shaded from green (open) to crimson (blocked).*

## 5. Quickstart & Open-Source Integration
Built in pure Python with zero proprietary dependencies, formatted as a modular addition for `databallpy.features.defensive_lanes`.
```bash
git clone [https://github.com/lukaspobisml/analytics_cup_2.0.git](https://github.com/lukaspobisml/analytics_cup_2.0.git)
cd analytics_cup_2.0
pip install -r requirements.txt
python -c "from src.features.lane_occlusion import DefensiveLaneOcclusion; print('DLOI Engine Ready')"
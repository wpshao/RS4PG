# RS4PG: Rolling Shutter Correction for Photogrammetry

<p align="center">
Wanpeng Shao†, Muhua Zhu†*, Letian Cao Yifei Xue, Tie Ji, Yizhen Lao*
</p>

<div align="center">

## 🚀 Features
- A pixel-wise NL-RSC solver for accurate modeling and correction of nonlinear rolling shutter distortions using local feature correspondences.
- A lightweight RSC4PG plugin for seamless integration into photogrammetry pipelines, supporting various consumer-grade RS cameras (UAVs and handheld devices).

## Installation

```bash
Clone the repository and run the demo files located in the NLRSC folder. Optical flow support is powered by open-mmlab/mmflow; you must install it first by following their official installation guide. Finally, install all remaining Python dependencies from the requirements.txt file.
```

### Usage
The `solver` is the core of the `NLRSC`, using quadratic models for fast rolling shutter correction, which receives the optical flow fields and returns the correction field. The `feats_sampling` function warps the RS image back to GS one.

<details>
<summary>0. Hyparameters </summary>

- `gamma`: Readout ratio γ (scanning time per row).
- `tau`: Target timestamp τ to warp to (0 for GS frame at t=0).

</details>


<details>
<summary>Quadratic rolling shutter correction</summary>
<p>
Quadratic_flow receives two optical flow fields from I₀ → I₋₁ and I₀ → I₁, and returns a correction field D_corr which rectifies the rolling shutter frame to the global shutter frame.
```
</p>
</details>


## 🍀 Demo
We provided a demo for rolling shutter correction in `NLRSC` based on close-range photogrammetry pics， which read the images from the `demo` folder and save the results in the `out` folder. Other datasets are stored in RS-Photogrammetry Dataset

```bash
python -m rspy/demo.py 
```
You can also use your own dataset with a suitable `gamma` and `tau` to get a satisfactory result.


### RS-Photogrammetry Dataset
```
通过网盘分享的文件：RS-Photogrammetry-Dataset
链接: https://pan.baidu.com/s/1gEsO3q90CWK9AIxHZATmuw 提取码: h27a 
```




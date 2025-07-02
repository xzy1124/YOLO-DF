# 🐟 UniFish Dataset

> A large-scale underwater fish detection dataset for real-world deployment and research.

---

## 📦 Download | 数据集下载

The full UniFish dataset can be downloaded from:

- [🔗 Google Drive (1.28 GB)]([https://drive.google.com/your-link-here](https://drive.google.com/file/d/17h1jRff5gCIjqREsDoRFOJck4kU8lyVs/view?usp=sharing))
- [🔗 Zenodo (with DOI)](https://doi.org/10.xxxx/zenodo.xxxx)  *(if available)*

完整数据集下载链接如下：

- [🔗 Google Drive（1.28 GB）]([https://drive.google.com/your-link-here](https://drive.google.com/file/d/17h1jRff5gCIjqREsDoRFOJck4kU8lyVs/view?usp=sharing))
- [🔗 Zenodo（带 DOI）](https://doi.org/10.xxxx/zenodo.xxxx) （如已上传）

---

## 📝 Dataset Description | 数据集简介

UniFish is a curated underwater fish detection dataset containing:

- 10,880 images from real underwater scenarios (BRUV, ROV, fixed cameras, etc.)
- 31 fish categories
- YOLO-style annotations (`.txt`) for object detection tasks

UniFish 是一个面向鱼类目标检测的大规模水下图像数据集，具有以下特点：

- 共包含 10,880 张图像，来自真实的水下环境（如 BRUV、ROV、固定摄像头等）
- 标注了 31 种鱼类类别
- 提供 YOLO 格式的目标检测标注（`.txt`）

---

## 🗂 File Structure | 文件结构

```plaintext
UniFish/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   └── train/
│   ├── val/


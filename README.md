# 🐟 UniFish Dataset

> A large-scale underwater fish detection dataset for real-world deployment and research.

---

## 📦 Download | 数据集下载

The full UniFish dataset can be downloaded from:

- [🔗 Google Drive (1.2 GB)](https://drive.google.com/your-link-here)
- [🔗 Zenodo (with DOI)](https://doi.org/10.xxxx/zenodo.xxxx)  *(if available)*

完整数据集下载链接如下：

- [🔗 Google Drive（1.2 GB）](https://drive.google.com/your-link-here)
- [🔗 Zenodo（带 DOI）](https://doi.org/10.xxxx/zenodo.xxxx) （如已上传）

---

## 📝 Dataset Description | 数据集简介

UniFish is a curated underwater fish detection dataset containing:

- 11,000 images from real underwater scenarios (BRUV, ROV, fixed cameras, etc.)
- 31 fish categories (labeled by marine experts)
- COCO-style annotations (`.json`) for object detection tasks

UniFish 是一个面向鱼类目标检测的大规模水下图像数据集，具有以下特点：

- 共包含 11,000 张图像，来自真实的水下环境（如 BRUV、ROV、固定摄像头等）
- 标注了 31 种鱼类类别，全部由专家审核
- 提供 COCO 格式的目标检测标注（`annotations.json`）

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


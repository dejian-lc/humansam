
# HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.19924)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://dejian-lc.github.io/humansam/)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/dejian-lc/humansam)
[![ModelScope Dataset](https://img.shields.io/badge/ModelScope-Dataset-purple)](https://www.modelscope.cn/datasets/DawnOfDark/HFV)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/dejian-lc/humansam)
[![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple)](https://modelscope.cn/models/DawnOfDark/humansam)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 📖 Abstract

Numerous synthesized videos from generative models, especially human-centric ones that simulate realistic human actions, pose significant threats to human information security and authenticity. While progress has been made in binary forgery video detection, the lack of fine-grained understanding of forgery types raises concerns regarding both reliability and interpretability, which are critical for real-world applications. 

To address this limitation, we propose **HumanSAM**, a new framework that builds upon the fundamental challenges of video generation models. Specifically, HumanSAM aims to classify human-centric forgeries into three distinct types of artifacts commonly observed in generated content: **spatial**, **appearance**, and **motion anomaly**. 

To better capture the features of geometry, semantics and spatiotemporal consistency, we propose to generate the human forgery representation by fusing two branches of video understanding and spatial depth. We also adopt a rank-based confidence enhancement strategy during the training process to learn more robust representation by introducing three prior scores. For training and evaluation, we construct the first public benchmark, the **Human-centric Forgery Video (HFV)** dataset, with all types of forgeries carefully annotated semi-automatically. In our experiments, HumanSAM yields promising results in comparison with state-of-the-art methods, both in binary and multi-class forgery classification.

## 🚀 News
- **[2025-12-12]** The code and dataset are released.
- **[2025-06-26]** HumanSAM is accepted by ICCV 2025!

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dejian-lc/humansam.git
   cd humansam
   ```

2. **Download Checkpoints**
   
   Please download the `checkpoints` directory from [Hugging Face Model](https://huggingface.co/dejian-lc/humansam) or [ModelScope Model](https://modelscope.cn/models/DawnOfDark/humansam) and place it in the root directory. 
   
   > **Important:** This directory contains the necessary **Flash Attention wheels** required for the installation step below, as well as the pre-trained models.

3. **Create a conda environment**
   ```bash
   conda create -n internvideo_new python=3.10 -y
   conda activate internvideo_new
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Flash Attention (Highly Recommended)**

   Flash Attention is crucial for memory efficiency. Without it, GPU memory usage may **triple** (e.g., increasing from ~20GB to ~60GB). With the current configuration using Flash Attention, the model can be trained on a **24GB GPU**.

   We provide pre-compiled wheels compatible with **PyTorch 2.8.0** (Flash Attention 2.8.3) in the `checkpoints/flashattn_wheel` directory, which includes `flash_attn`, `fused_dense_lib`, and `layer_norm`.

   ```bash
   pip install checkpoints/flashattn_wheel/*.whl
   ```

   > **Note:** If you intend to use a different version of Flash Attention, you will need to clone the [Flash Attention repository](https://github.com/Dao-AILab/flash-attention) and manually compile `fused_dense_lib` and `layer_norm` using the CUDA Toolkit.

## 📂 Data Preparation

Please download the `data` directory from [Hugging Face Dataset](https://huggingface.co/datasets/dejian-lc/humansam) or [ModelScope Dataset](https://www.modelscope.cn/datasets/DawnOfDark/HFV) and place it in the root directory.

Organize the data structure as follows:

```
data/
├── train_data/
│   └── cls_test_82_win_video/
│       ├── [Video Files].mp4
│       ├── train_2.csv
│       ├── val_2.csv
│       ├── train_4.csv
│       └── val_4.csv
└── eval_data/
    ├── CogVideoX-5B_human_dim/
    │   ├── [Video Files].mp4
    │   ├── test_2.csv
    │   └── test_4.csv
    ├── Kling_human_dim/
    ├── MinMax_human_dim/
    ├── Vchitect-2.0-2B_human_dim/
    ├── Vchitect-2.0-2B+VEnhancer_human_dim/
    ├── gen_2_human_dim/
    ├── gen3_human_dim/
    └── pika_1_human_dim/
```

### Custom Dataset Organization

If you organize your own dataset, please ensure it follows the structure above.

- **File Naming**:
  - For **binary classification**, use the `_2` suffix for CSV files (e.g., `train_2.csv`, `val_2.csv`, `test_2.csv`).
  - For **multi-class classification** (4 classes), use the `_4` suffix for CSV files (e.g., `train_4.csv`, `val_4.csv`, `test_4.csv`).

- **CSV Content Format**:
  - **Column 1**: Video filename.
  - **Column 2**: Corresponding label.
  - **Column 3** (Optional for 2-class or 4-class training CSV): Confidence score.

## 🏋️ Training

To train the HumanSAM model on the HFV dataset, please use the provided `run_accelerate.sh` script:

```bash
bash run_accelerate.sh
```

**Key Parameters:**
- `nb_classes`: Specify the number of classification classes (e.g., `2` for binary classification, `4` for multi-class classification).
- `--use_conf`: Add this flag to enable the rank-based confidence enhancement strategy (requires confidence scores in the training CSV).

> **Note:** Please adjust the hyperparameters in `run_accelerate.sh` or via command line arguments according to your hardware configuration.

## 🧪 Evaluation

The `checkpoints/humansam` directory contains pre-trained models for both **binary** and **4-class** evaluation on the HFV dataset.

To evaluate the model on the test set:

```bash
# Single dataset evaluation
bash run_eval.sh

# Batch evaluation for multiple datasets
bash batch_eval.sh
```

## 🤝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2025humansam,
  title={HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly},
  author={Liu, Chang and Ye, Yunfan and Zhang, Fan and Zhou, Qingyang and Luo, Yuchuan and Cai, Zhiping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14028--14038},
  year={2025}
}
```

## 🙏 Acknowledgement

This project is built upon the following open-source projects. We thank the authors for their great work!
- [InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2)
- [Depth Pro](https://github.com/apple/ml-depth-pro)

## 📧 Contact

If you have any questions, please feel free to contact us at liudawn@nudt.edu.cn or open an issue in this repository.


# HumanSAM: Classifying Human-centric Forgery Videos in Human Spatial, Appearance, and Motion Anomaly

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/[LINK])
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://dejian-lc.github.io/humansam/)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/[LINK])
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/[LINK])
[![ModelScope](https://img.shields.io/badge/ModelScope-Model-purple)](https://modelscope.cn/models/[LINK])
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## 📖 Abstract

Numerous synthesized videos from generative models, especially human-centric ones that simulate realistic human actions, pose significant threats to human information security and authenticity. While progress has been made in binary forgery video detection, the lack of fine-grained understanding of forgery types raises concerns regarding both reliability and interpretability, which are critical for real-world applications. 

To address this limitation, we propose **HumanSAM**, a new framework that builds upon the fundamental challenges of video generation models. Specifically, HumanSAM aims to classify human-centric forgeries into three distinct types of artifacts commonly observed in generated content: **spatial**, **appearance**, and **motion anomaly**. 

To better capture the features of geometry, semantics and spatiotemporal consistency, we propose to generate the human forgery representation by fusing two branches of video understanding and spatial depth. We also adopt a rank-based confidence enhancement strategy during the training process to learn more robust representation by introducing three prior scores. For training and evaluation, we construct the first public benchmark, the **Human-centric Forgery Video (HFV)** dataset, with all types of forgeries carefully annotated semi-automatically. In our experiments, HumanSAM yields promising results in comparison with state-of-the-art methods, both in binary and multi-class forgery classification.

## 🚀 News
- **[2025-12-12]** The code and dataset are released.
- **[2025-XX-XX]** HumanSAM is accepted by ICCV 2025!

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dejian-lc/humansam.git
   cd humansam
   ```

2. **Create a conda environment**
   ```bash
   conda create -n internvideo_new python=3.10 -y
   conda activate internvideo_new
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Flash Attention**
   ```bash
   # Ensure you have the correct wheel file in the checkpoints directory
   pip install checkpoints/flashattn_wheel/*.whl
   ```

## 📂 Data Preparation

Please download the **HFV Dataset** from [HuggingFace](https://huggingface.co/datasets/[LINK]) or [ModelScope](https://modelscope.cn/datasets/[LINK]).

Organize the data structure as follows:

```
data/
├── train_data/
│   ├── spatial/
│   ├── appearance/
│   └── motion/
└── eval_data/
    ├── spatial/
    ├── appearance/
    └── motion/
```

## 🏋️ Training

To train the HumanSAM model on the HFV dataset, run the following command:

```bash
# Example training command
python run_finetuning_combine.py \
    --model humansam \
    --data_path ./data/train_data \
    --output_dir ./checkpoints/humansam_finetuned \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4
```

> **Note:** Please adjust the hyperparameters in `run_finetuning_combine.py` or via command line arguments according to your hardware configuration.

## 🧪 Evaluation

To evaluate the trained model on the test set:

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
  author={Liu, Chang and [Other Authors]},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 📧 Contact

If you have any questions, please feel free to contact us at [EMAIL] or open an issue in this repository.

# Rethinking the Temperature for Federated Heterogeneous Distillation
<font style="color:rgb(31, 35, 40);">An official implementation for "</font>[Rethinking the Temperature for Federated Heterogeneous Distillation](https://openreview.net/pdf?id=f9xsNQ8oSd)", ICML2025.

# <font style="color:rgb(31, 35, 40);">Installation</font>
```bash
conda create -n ReT-FHD
conda activate ReT-FHD
pip install -r requirements.txt
```

# Data Preparation
<font style="color:rgb(31, 35, 40);">After installation, prepare the dataset according to the following instructions.</font>

<font style="color:rgb(31, 35, 40);">Please run the script</font><font style="color:rgb(31, 35, 40);">:</font>

```bash
python dataset/cifar.py noniid x dir
```

# <font style="color:rgb(31, 35, 40);">Training</font>
```bash
cd ReT-FHD
python run.py --dataset /ptah/to/CIFAR_10 --num_classes 10 --max_rounds 400
```

# <font style="color:rgb(31, 35, 40);">Citing our work</font>
<font style="color:rgb(31, 35, 40);">If you find this paper useful, please consider staring this repo and citing our paper!</font>

```bash
@inproceedings{qi2025rethinking,
  title={Rethinking the Temperature for Federated Heterogeneous Distillation},
  author={Fan Qi and Daxu Shi and Chuokun Xu and Shuai Li and Changsheng Xu},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```




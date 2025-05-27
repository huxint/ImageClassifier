# 图像分类实验

本项目实现了一个图像分类系统，使用多种特征提取方法和分类器对图像进行分类。

## 项目结构

```
.
├── feature_extractors.py  # 特征提取器实现
├── classifiers.py         # 分类器实现
├── utils.py              # 工具函数
├── main.py               # 主程序
├── requirements.txt      # 依赖项
└── README.md             # 说明文档
```

## 环境要求

- Python 3.9+
- CUDA (可选，用于加速CNN特征提取)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集说明

数据集包含20个类别共2000张图片，其中：
- 训练集：每个类别50张图片，共1000张
- 测试集：每个类别50张图片，共1000张

图片按照以下编号规则组织：
- 类别0：0-49(训练), 50-99(测试)
- 类别1：100-149(训练), 150-199(测试)
- ...
- 类别19：1900-1949(训练), 1950-1999(测试)

## 特征提取

本项目实现了以下特征提取方法：

1. **灰度共生矩特征 (GLCM)**：描述纹理特征
2. **SIFT特征 + BOW方法**：提取局部特征并使用词袋模型
3. **局部二值模式 (LBP)**：提取纹理特征
4. **颜色特征**：包括颜色直方图和颜色矩
5. **CNN特征**：使用预训练的ResNet18模型提取深度特征

## 分类方法

本项目实现了以下分类器：

1. **SVM**：支持向量机，可使用不同核函数
2. **随机森林**：集成学习方法
3. **K近邻**：基于距离的分类方法
4. **集成分类器**：组合多个基分类器的结果

## 使用方法

### 基本用法

```bash
python main.py --data_dir dataset --output_dir output
```

```bash
python main.py --load_features --features glcm sift_bow lbp color cnn --fusion weighted
```

```bash
python main.py --load_features --features glcm sift_bow lbp color cnn
```

```bash
python main.py --load_features --features glcm sift_bow lbp color cnn --classifier rf
```

### 高级用法

```bash
# 指定特征类型
python main.py --features glcm sift_bow lbp color cnn

# 指定分类器
python main.py --classifier svm  # 可选: svm, rf, knn, ensemble

# 指定特征融合方法
python main.py --fusion weighted  # 可选: concat, weighted

# 加载已保存的特征
python main.py --load_features
```

### 参数说明

- `--data_dir`：数据集目录，默认为`dataset`
- `--output_dir`：输出目录，默认为`output`
- `--features`：要提取的特征类型，可多选
- `--classifier`：分类器类型，默认为`svm`
- `--fusion`：特征融合方法，默认为`concat`
- `--load_features`：是否加载已保存的特征

## 输出结果

程序会在输出目录中生成以下文件：

1. `{classifier}_predictions.txt`：预测结果
2. `{classifier}_class_accuracies.txt`：每个类别的准确率
3. `{classifier}_confusion_matrix.png`：混淆矩阵可视化
4. `features/`：保存提取的特征

## 注意事项

- CNN特征提取需要较大的内存和GPU资源
- 首次运行时需要下载预训练模型
- 对于大数据集，建议先提取特征并保存，然后使用`--load_features`参数加载特征 
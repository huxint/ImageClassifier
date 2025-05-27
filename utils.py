import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

def get_image_paths_and_labels(data_dir, start_idx=0, end_idx=1999):
    """
    获取图像路径和标签
    
    参数:
    data_dir: 数据集目录
    start_idx: 起始索引
    end_idx: 结束索引
    
    返回:
    image_paths: 图像路径列表
    labels: 标签列表
    """
    image_paths = []
    labels = []
    
    for idx in range(start_idx, end_idx + 1):
        image_path = os.path.join(data_dir, f"{idx}.jpg")
        if os.path.exists(image_path):
            image_paths.append(image_path)
            # 根据索引确定类别
            label = idx // 100
            labels.append(label)
    
    return image_paths, labels

def get_train_test_data(data_dir):
    """
    获取训练集和测试集数据
    
    参数:
    data_dir: 数据集目录
    
    返回:
    train_paths: 训练集图像路径
    train_labels: 训练集标签
    test_paths: 测试集图像路径
    test_labels: 测试集标签
    """
    # 训练集: 每个类别的前50张图片 (0-49, 100-149, ..., 1900-1949)
    train_paths = []
    train_labels = []
    
    # 测试集: 每个类别的后50张图片 (50-99, 150-199, ..., 1950-1999)
    test_paths = []
    test_labels = []
    
    for class_idx in range(20):
        # 训练集
        start_idx = class_idx * 100
        end_idx = start_idx + 49
        paths, labels = get_image_paths_and_labels(data_dir, start_idx, end_idx)
        train_paths.extend(paths)
        train_labels.extend(labels)
        
        # 测试集
        start_idx = class_idx * 100 + 50
        end_idx = start_idx + 49
        paths, labels = get_image_paths_and_labels(data_dir, start_idx, end_idx)
        test_paths.extend(paths)
        test_labels.extend(labels)
    
    return train_paths, train_labels, test_paths, test_labels

def save_results(predictions, true_labels, output_file):
    """
    保存预测结果到文件
    
    参数:
    predictions: 预测标签
    true_labels: 真实标签
    output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            # 图片序号 真实类别 预测类别
            image_idx = 50 + (true * 100) + (i % 50)
            f.write(f"{image_idx} {true} {pred}\n")

def save_class_accuracies(class_accuracies, output_file):
    """
    保存每个类别的准确率到文件
    
    参数:
    class_accuracies: 每个类别的准确率列表
    output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, acc in enumerate(class_accuracies):
            f.write(f"类别 {i}: {acc:.4f}\n")
        f.write(f"平均准确率: {np.mean(class_accuracies):.4f}\n")

def plot_confusion_matrix(conf_matrix, class_names=None, output_file=None):
    """
    绘制混淆矩阵
    
    参数:
    conf_matrix: 混淆矩阵
    class_names: 类别名称
    output_file: 输出文件路径
    """
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，可能会导致中文显示问题")
    
    # 归一化混淆矩阵
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(classifier, feature_names, output_file=None):
    """
    绘制特征重要性（仅适用于随机森林等具有feature_importances_属性的分类器）
    
    参数:
    classifier: 分类器
    feature_names: 特征名称列表
    output_file: 输出文件路径
    """
    if hasattr(classifier.model, 'feature_importances_'):
        # 获取特征重要性
        importances = classifier.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        plt.title('特征重要性')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("该分类器不支持特征重要性分析") 
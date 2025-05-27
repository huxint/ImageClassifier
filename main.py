import os
import numpy as np
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from feature_extractors import (
    GLCMFeatureExtractor, 
    SIFTBOWFeatureExtractor, 
    LBPFeatureExtractor, 
    ColorFeatureExtractor, 
    CNNFeatureExtractor,
    FeatureFusion
)
from classifiers import SVMClassifier, RandomForestClassifier_, KNNClassifier, EnsembleClassifier
from utils import get_train_test_data, save_results, plot_confusion_matrix, save_class_accuracies

def extract_features(train_paths, test_paths, feature_types, output_dir):
    """
    提取特征并保存
    
    参数:
    train_paths: 训练集图像路径
    test_paths: 测试集图像路径
    feature_types: 特征类型列表
    output_dir: 输出目录
    
    返回:
    train_features: 训练集特征
    test_features: 测试集特征
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_features_all = {}
    test_features_all = {}
    
    # 提取各种特征
    for feature_type in feature_types:
        print(f"提取{feature_type}特征...")
        
        # 根据特征类型创建提取器
        if feature_type == 'glcm':
            extractor = GLCMFeatureExtractor()
        elif feature_type == 'sift_bow':
            extractor = SIFTBOWFeatureExtractor(n_clusters=100)
            # 构建词汇表
            print("构建SIFT词汇表...")
            extractor.build_vocabulary(train_paths, sample_size=500)
        elif feature_type == 'lbp':
            extractor = LBPFeatureExtractor()
        elif feature_type == 'color':
            extractor = ColorFeatureExtractor()
        elif feature_type == 'cnn':
            extractor = CNNFeatureExtractor(model_name='resnet18')
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")
        
        # 提取训练集特征
        print("提取训练集特征...")
        train_features = extractor.extract_batch(train_paths)
        
        # 提取测试集特征
        print("提取测试集特征...")
        test_features = extractor.extract_batch(test_paths)
        
        # 保存特征
        np.save(os.path.join(output_dir, f"train_{feature_type}_features.npy"), train_features)
        np.save(os.path.join(output_dir, f"test_{feature_type}_features.npy"), test_features)
        
        train_features_all[feature_type] = train_features
        test_features_all[feature_type] = test_features
        
        print(f"{feature_type}特征形状: {train_features.shape}")
    
    return train_features_all, test_features_all

def load_features(feature_types, features_dir):
    """
    加载特征
    
    参数:
    feature_types: 特征类型列表
    features_dir: 特征目录
    
    返回:
    train_features: 训练集特征
    test_features: 测试集特征
    """
    train_features_all = {}
    test_features_all = {}
    
    for feature_type in feature_types:
        train_features_path = os.path.join(features_dir, f"train_{feature_type}_features.npy")
        test_features_path = os.path.join(features_dir, f"test_{feature_type}_features.npy")
        
        if os.path.exists(train_features_path) and os.path.exists(test_features_path):
            train_features_all[feature_type] = np.load(train_features_path)
            test_features_all[feature_type] = np.load(test_features_path)
            print(f"已加载{feature_type}特征，形状: {train_features_all[feature_type].shape}")
        else:
            print(f"警告: 未找到{feature_type}特征文件")
    
    return train_features_all, test_features_all

def fuse_features(train_features_all, test_features_all, fusion_method='concat'):
    """
    融合特征
    
    参数:
    train_features_all: 所有训练集特征
    test_features_all: 所有测试集特征
    fusion_method: 融合方法，'concat'或'weighted'
    
    返回:
    fused_train_features: 融合后的训练集特征
    fused_test_features: 融合后的测试集特征
    """
    if fusion_method == 'concat':
        # 简单拼接
        fused_train_features = np.hstack([train_features_all[ft] for ft in train_features_all])
        fused_test_features = np.hstack([test_features_all[ft] for ft in test_features_all])
    elif fusion_method == 'weighted':
        # 加权融合（需要先归一化特征）
        weights = {
            'glcm': 1.1,
            'sift_bow': 1.0,
            'lbp': 1.0,
            'color': 1.0,
            'cnn': 5.0  # CNN特征权重更高
        }
        
        fused_train_features = []
        fused_test_features = []
        
        for ft in train_features_all:
            # 归一化，避免除以0
            train_norms = np.linalg.norm(train_features_all[ft], axis=1, keepdims=True)
            test_norms = np.linalg.norm(test_features_all[ft], axis=1, keepdims=True)
            
            # 防止除以0，将0范数替换为1
            train_norms[train_norms == 0] = 1.0
            test_norms[test_norms == 0] = 1.0
            
            train_norm = train_features_all[ft] / train_norms
            test_norm = test_features_all[ft] / test_norms
            
            # 加权
            weight = weights.get(ft, 1.0)
            fused_train_features.append(train_norm * weight)
            fused_test_features.append(test_norm * weight)
        
        fused_train_features = np.hstack(fused_train_features)
        fused_test_features = np.hstack(fused_test_features)
    else:
        raise ValueError(f"不支持的融合方法: {fusion_method}")
    
    print(f"融合后特征形状: {fused_train_features.shape}")
    return fused_train_features, fused_test_features

def train_and_evaluate(train_features, train_labels, test_features, test_labels, classifier_type, output_dir):
    """
    训练分类器并评估
    
    参数:
    train_features: 训练集特征
    train_labels: 训练集标签
    test_features: 测试集特征
    test_labels: 测试集标签
    classifier_type: 分类器类型
    output_dir: 输出目录
    
    返回:
    results: 评估结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分类器
    if classifier_type == 'svm':
        classifier = SVMClassifier(kernel='rbf', C=10.0)
    elif classifier_type == 'rf':
        classifier = RandomForestClassifier_(n_estimators=200, max_depth=None)
    elif classifier_type == 'knn':
        classifier = KNNClassifier(n_neighbors=5)
    elif classifier_type == 'ensemble':
        # 创建多个基分类器
        svm = SVMClassifier(kernel='rbf', C=10.0)
        rf = RandomForestClassifier_(n_estimators=200, max_depth=None)
        knn = KNNClassifier(n_neighbors=5)
        
        # 创建集成分类器
        classifier = EnsembleClassifier([svm, rf, knn], weights=[1.5, 1.0, 0.8])
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")
    
    # 训练分类器
    print(f"训练{classifier_type}分类器...")
    start_time = time.time()
    classifier.train(train_features, train_labels)
    train_time = time.time() - start_time
    print(f"训练耗时: {train_time:.2f}秒")
    
    # 评估分类器
    print("评估分类器...")
    results = classifier.evaluate(test_features, test_labels)
    
    # 保存结果
    print(f"平均准确率: {results['accuracy']:.4f}")
    
    # 保存预测结果
    save_results(results['predictions'], test_labels, 
                os.path.join(output_dir, f"{classifier_type}_predictions.txt"))
    
    # 保存类别准确率
    save_class_accuracies(results['class_accuracies'], 
                         os.path.join(output_dir, f"{classifier_type}_class_accuracies.txt"))
    
    # 绘制混淆矩阵
    class_names = [str(i) for i in range(20)]
    plot_confusion_matrix(results['confusion_matrix'], class_names,
                         os.path.join(output_dir, f"{classifier_type}_confusion_matrix.png"))
    
    # 保存模型
    classifier.save(os.path.join(output_dir, f"{classifier_type}_model.pkl"))
    
    return results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图像分类实验")
    parser.add_argument('--data_dir', type=str, default='dataset', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['glcm', 'sift_bow', 'lbp', 'color', 'cnn'], 
                        help='要提取的特征类型')
    parser.add_argument('--classifier', type=str, default='svm', 
                        choices=['svm', 'rf', 'knn', 'ensemble'], 
                        help='分类器类型')
    parser.add_argument('--fusion', type=str, default='concat', 
                        choices=['concat', 'weighted'], 
                        help='特征融合方法')
    parser.add_argument('--load_features', action='store_true', 
                        help='是否加载已保存的特征')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    features_dir = os.path.join(args.output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    # 获取训练集和测试集
    print("加载数据集...")
    train_paths, train_labels, test_paths, test_labels = get_train_test_data(args.data_dir)
    print(f"训练集: {len(train_paths)}张图片")
    print(f"测试集: {len(test_paths)}张图片")
    
    # 提取或加载特征
    if args.load_features:
        print("加载已保存的特征...")
        train_features_all, test_features_all = load_features(args.features, features_dir)
    else:
        print("提取特征...")
        train_features_all, test_features_all = extract_features(
            train_paths, test_paths, args.features, features_dir)
    
    # 融合特征
    print(f"使用{args.fusion}方法融合特征...")
    train_features, test_features = fuse_features(
        train_features_all, test_features_all, args.fusion)
    
    # 训练和评估分类器
    print(f"使用{args.classifier}分类器...")
    results = train_and_evaluate(
        train_features, train_labels, test_features, test_labels, 
        args.classifier, args.output_dir)
    
    print("实验完成！")
    print(f"平均准确率: {results['accuracy']:.4f}")

if __name__ == '__main__':
    main() 
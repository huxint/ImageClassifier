import os
import numpy as np
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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


class ImageClassifier:
    """图像分类系统主类"""
    
    def __init__(self, args):
        """
        初始化图像分类系统
        
        参数:
        args: 命令行参数
        """
        self.args = args
        self.train_paths = None
        self.train_labels = None
        self.test_paths = None
        self.test_labels = None
        self.train_features_all = {}
        self.test_features_all = {}
        self.train_features = None
        self.test_features = None
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        self.features_dir = os.path.join(args.output_dir, 'features')
        os.makedirs(self.features_dir, exist_ok=True)
    
    def load_dataset(self):
        """加载数据集"""
        print("加载数据集...")
        self.train_paths, self.train_labels, self.test_paths, self.test_labels = get_train_test_data(self.args.data_dir)
        print(f"训练集: {len(self.train_paths)}张图片")
        print(f"测试集: {len(self.test_paths)}张图片")
    
    def extract_features(self):
        """提取特征"""
        if self.args.load_features:
            print("加载已保存的特征...")
            self.train_features_all, self.test_features_all = self._load_features()
        else:
            print("提取特征...")
            self.train_features_all, self.test_features_all = self._extract_features()
        
        # 融合特征
        print(f"使用{self.args.fusion}方法融合特征...")
        self.train_features, self.test_features = self._fuse_features()
    
    def _extract_features(self):
        """提取特征并保存"""
        train_features_all = {}
        test_features_all = {}
        
        # 提取各种特征
        for feature_type in self.args.features:
            print(f"提取{feature_type}特征...")
            
            # 根据特征类型创建提取器
            if feature_type == 'glcm':
                extractor = GLCMFeatureExtractor()
            elif feature_type == 'sift_bow':
                extractor = SIFTBOWFeatureExtractor(n_clusters=100)
                # 构建词汇表
                print("构建SIFT词汇表...")
                extractor.build_vocabulary(self.train_paths, sample_size=500)
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
            train_features = extractor.extract_batch(self.train_paths)
            
            # 提取测试集特征
            print("提取测试集特征...")
            test_features = extractor.extract_batch(self.test_paths)
            
            # 保存特征
            np.save(os.path.join(self.features_dir, f"train_{feature_type}_features.npy"), train_features)
            np.save(os.path.join(self.features_dir, f"test_{feature_type}_features.npy"), test_features)
            
            train_features_all[feature_type] = train_features
            test_features_all[feature_type] = test_features
            
            print(f"{feature_type}特征形状: {train_features.shape}")
        
        return train_features_all, test_features_all
    
    def _load_features(self):
        """加载已保存的特征"""
        train_features_all = {}
        test_features_all = {}
        
        for feature_type in self.args.features:
            train_features_path = os.path.join(self.features_dir, f"train_{feature_type}_features.npy")
            test_features_path = os.path.join(self.features_dir, f"test_{feature_type}_features.npy")
            
            if os.path.exists(train_features_path) and os.path.exists(test_features_path):
                train_features_all[feature_type] = np.load(train_features_path)
                test_features_all[feature_type] = np.load(test_features_path)
                print(f"已加载{feature_type}特征，形状: {train_features_all[feature_type].shape}")
            else:
                print(f"警告: 未找到{feature_type}特征文件")
        
        return train_features_all, test_features_all
    
    def _fuse_features(self):
        """融合特征"""
        if self.args.fusion == 'concat':
            # 简单拼接
            fused_train_features = np.hstack([self.train_features_all[ft] for ft in self.train_features_all])
            fused_test_features = np.hstack([self.test_features_all[ft] for ft in self.test_features_all])
        elif self.args.fusion == 'weighted':
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
            
            for ft in self.train_features_all:
                # 归一化，避免除以0
                train_norms = np.linalg.norm(self.train_features_all[ft], axis=1, keepdims=True)
                test_norms = np.linalg.norm(self.test_features_all[ft], axis=1, keepdims=True)
                
                # 防止除以0，将0范数替换为1
                train_norms[train_norms == 0] = 1.0
                test_norms[test_norms == 0] = 1.0
                
                train_norm = self.train_features_all[ft] / train_norms
                test_norm = self.test_features_all[ft] / test_norms
                
                # 加权
                weight = weights.get(ft, 1.0)
                fused_train_features.append(train_norm * weight)
                fused_test_features.append(test_norm * weight)
            
            fused_train_features = np.hstack(fused_train_features)
            fused_test_features = np.hstack(fused_test_features)
        else:
            raise ValueError(f"不支持的融合方法: {self.args.fusion}")
        
        print(f"融合后特征形状: {fused_train_features.shape}")
        return fused_train_features, fused_test_features
    
    def train_and_evaluate(self):
        """训练分类器并评估"""
        print(f"使用{self.args.classifier}分类器...")
        
        # 创建分类器
        classifier = self._create_classifier()
        
        # 训练分类器
        print(f"训练{self.args.classifier}分类器...")
        start_time = time.time()
        classifier.train(self.train_features, self.train_labels)
        train_time = time.time() - start_time
        print(f"训练耗时: {train_time:.2f}秒")
        
        # 评估分类器
        print("评估分类器...")
        results = classifier.evaluate(self.test_features, self.test_labels)
        
        # 保存结果
        print(f"平均准确率: {results['accuracy']:.4f}")
        
        # 保存预测结果
        save_results(results['predictions'], self.test_labels, 
                    os.path.join(self.args.output_dir, f"{self.args.classifier}_predictions.txt"))
        
        # 保存类别准确率
        save_class_accuracies(results['class_accuracies'], 
                             os.path.join(self.args.output_dir, f"{self.args.classifier}_class_accuracies.txt"))
        
        # 绘制混淆矩阵
        class_names = [str(i) for i in range(20)]
        plot_confusion_matrix(results['confusion_matrix'], class_names,
                             os.path.join(self.args.output_dir, f"{self.args.classifier}_confusion_matrix.png"))
        
        # 保存模型
        classifier.save(os.path.join(self.args.output_dir, f"{self.args.classifier}_model.pkl"))
        
        return results
    
    def _create_classifier(self):
        """创建分类器"""
        if self.args.classifier == 'svm':
            if self.args.use_cv:
                print("使用交叉验证优化SVM参数...")
                classifier = SVMClassifier()
                best_params = classifier.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
                print(f"最佳参数: {best_params}")
            else:
                classifier = SVMClassifier(kernel='rbf', C=10.0)
        elif self.args.classifier == 'rf':
            if self.args.use_cv:
                print("使用交叉验证优化随机森林参数...")
                classifier = RandomForestClassifier_()
                best_params = classifier.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
                print(f"最佳参数: {best_params}")
            else:
                classifier = RandomForestClassifier_(n_estimators=200, max_depth=None)
        elif self.args.classifier == 'knn':
            if self.args.use_cv:
                print("使用交叉验证优化KNN参数...")
                classifier = KNNClassifier()
                best_params = classifier.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
                print(f"最佳参数: {best_params}")
            else:
                classifier = KNNClassifier(n_neighbors=5)
        elif self.args.classifier == 'ensemble':
            # 创建多个基分类器
            if self.args.use_cv:
                print("使用交叉验证优化集成分类器参数...")
                svm = SVMClassifier()
                svm.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
                
                rf = RandomForestClassifier_()
                rf.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
                
                knn = KNNClassifier()
                knn.tune_hyperparameters(self.train_features, self.train_labels, cv=self.args.cv_folds)
            else:
                svm = SVMClassifier(kernel='rbf', C=10.0)
                rf = RandomForestClassifier_(n_estimators=200, max_depth=None)
                knn = KNNClassifier(n_neighbors=5)
            
            # 创建集成分类器
            classifier = EnsembleClassifier([svm, rf, knn], weights=[1.5, 1.0, 0.8])
        else:
            raise ValueError(f"不支持的分类器类型: {self.args.classifier}")
        
        return classifier
    
    def run(self):
        """运行完整的图像分类流程"""
        # 1. 加载数据集
        self.load_dataset()
        
        # 2. 提取特征
        self.extract_features()
        
        # 3. 训练和评估分类器
        results = self.train_and_evaluate()
        
        print("实验完成！")
        print(f"平均准确率: {results['accuracy']:.4f}")
        
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
    parser.add_argument('--use_cv', action='store_true',
                        help='是否使用交叉验证优化参数')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建图像分类器实例
    classifier = ImageClassifier(args)
    
    # 运行图像分类流程
    classifier.run()


if __name__ == '__main__':
    main() 
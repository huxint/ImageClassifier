import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

class BaseClassifier:
    """分类器基类"""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        """训练分类器"""
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        # 训练模型
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        """预测类别"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        # 预测
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 计算每个类别的准确率
        class_accuracies = []
        for i in range(len(conf_matrix)):
            if np.sum(conf_matrix[i]) > 0:
                class_accuracies.append(conf_matrix[i, i] / np.sum(conf_matrix[i]))
            else:
                class_accuracies.append(0)
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        
    def load(self, path):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']

class SVMClassifier(BaseClassifier):
    """SVM分类器"""
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        super().__init__("SVM")
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        
    def tune_hyperparameters(self, X, y, cv=3):
        """调整超参数"""
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # 使用网格搜索找到最佳参数
        X_scaled = self.scaler.fit_transform(X)
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        # 更新模型
        self.model = grid_search.best_estimator_
        print(f"最佳SVM参数: {grid_search.best_params_}")
        
        return grid_search.best_params_

class RandomForestClassifier_(BaseClassifier):
    """随机森林分类器"""
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
    def tune_hyperparameters(self, X, y, cv=3):
        """调整超参数"""
        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # 使用网格搜索找到最佳参数
        X_scaled = self.scaler.fit_transform(X)
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        # 更新模型
        self.model = grid_search.best_estimator_
        print(f"最佳随机森林参数: {grid_search.best_params_}")
        
        return grid_search.best_params_

class KNNClassifier(BaseClassifier):
    """K近邻分类器"""
    def __init__(self, n_neighbors=5):
        super().__init__("KNN")
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    def tune_hyperparameters(self, X, y, cv=3):
        """调整超参数"""
        # 定义参数网格
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        # 使用网格搜索找到最佳参数
        X_scaled = self.scaler.fit_transform(X)
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        # 更新模型
        self.model = grid_search.best_estimator_
        print(f"最佳KNN参数: {grid_search.best_params_}")
        
        return grid_search.best_params_

class EnsembleClassifier(BaseClassifier):
    """集成分类器"""
    def __init__(self, classifiers, weights=None):
        """
        初始化集成分类器
        
        参数:
        classifiers: 分类器列表
        weights: 各分类器权重，默认为均等权重
        """
        super().__init__("Ensemble")
        self.classifiers = classifiers
        
        if weights is None:
            self.weights = np.ones(len(classifiers)) / len(classifiers)
        else:
            self.weights = np.array(weights) / np.sum(weights)
    
    def train(self, X, y):
        """训练所有基分类器"""
        for classifier in self.classifiers:
            classifier.train(X, y)
    
    def predict(self, X):
        """通过投票进行预测"""
        # 获取每个分类器的预测结果
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict(X))
        
        # 加权投票
        final_predictions = []
        for i in range(len(X)):
            votes = {}
            for j, pred in enumerate(predictions):
                vote = pred[i]
                if vote not in votes:
                    votes[vote] = 0
                votes[vote] += self.weights[j]
            
            # 选择得票最高的类别
            final_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
        
        return np.array(final_predictions)
    
    def save(self, path):
        """保存所有基分类器"""
        os.makedirs(path, exist_ok=True)
        for i, classifier in enumerate(self.classifiers):
            classifier.save(os.path.join(path, f"{classifier.name}_{i}.pkl"))
        # 保存权重
        np.save(os.path.join(path, "weights.npy"), self.weights)
        
    def load(self, path):
        """加载所有基分类器"""
        for i, classifier in enumerate(self.classifiers):
            classifier.load(os.path.join(path, f"{classifier.name}_{i}.pkl"))
        # 加载权重
        self.weights = np.load(os.path.join(path, "weights.npy")) 
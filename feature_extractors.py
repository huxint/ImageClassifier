import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class FeatureExtractor:
    """特征提取基类"""
    def __init__(self):
        pass
    
    def extract(self, image_path):
        """提取特征的方法"""
        raise NotImplementedError("子类必须实现extract方法")
    
    def extract_batch(self, image_paths):
        """批量提取特征"""
        features = []
        for path in tqdm(image_paths, desc=f"提取{self.__class__.__name__}特征"):
            features.append(self.extract(path))
        return np.array(features)

class GLCMFeatureExtractor(FeatureExtractor):
    """灰度共生矩阵特征提取器"""
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
        super().__init__()
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    def extract(self, image_path):
        # 读取图像并转换为灰度图
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 将图像缩放到合适的大小，减少计算量
        gray = cv2.resize(gray, (128, 128))
        
        # 量化灰度级别
        bins = np.linspace(0, 255, self.levels+1)
        gray = np.digitize(gray, bins) - 1
        
        # 计算GLCM
        glcm = graycomatrix(gray, self.distances, self.angles, 
                            levels=self.levels, symmetric=True, normed=True)
        
        # 提取GLCM属性
        features = []
        for prop in self.properties:
            features.append(graycoprops(glcm, prop).flatten())
        
        return np.concatenate(features)

class SIFTBOWFeatureExtractor(FeatureExtractor):
    """SIFT特征 + BOW方法特征提取器"""
    def __init__(self, n_clusters=100):
        super().__init__()
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.scaler = StandardScaler()
        self.vocabulary = None
    
    def build_vocabulary(self, image_paths, sample_size=None):
        """构建词汇表"""
        if sample_size and sample_size < len(image_paths):
            # 随机采样部分图像来构建词汇表
            image_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        all_descriptors = []
        for path in tqdm(image_paths, desc="构建SIFT词汇表"):
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # 将所有描述子合并
        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
            # 使用KMeans聚类构建词汇表
            print(f"使用 {all_descriptors.shape[0]} 个SIFT描述子进行聚类...")
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(all_descriptors)
            self.vocabulary = self.kmeans.cluster_centers_
            print("词汇表构建完成")
        else:
            print("警告: 没有找到SIFT描述子")
    
    def extract(self, image_path):
        if self.kmeans is None:
            raise ValueError("请先调用build_vocabulary方法构建词汇表")
        
        # 读取图像并提取SIFT特征
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # 如果没有检测到关键点，返回零向量
        if descriptors is None:
            return np.zeros(self.n_clusters)
        
        # 计算词袋表示
        histogram = np.zeros(self.n_clusters)
        predictions = self.kmeans.predict(descriptors)
        for pred in predictions:
            histogram[pred] += 1
        
        # 归一化直方图
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
        
        return histogram

class LBPFeatureExtractor(FeatureExtractor):
    """局部二值模式(LBP)特征提取器"""
    def __init__(self, radius=3, n_points=24, method='uniform', n_bins=10):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.n_bins = n_bins
    
    def extract(self, image_path):
        # 读取图像并转换为灰度图
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算LBP特征
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        
        # 计算LBP直方图
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # 确保特征向量长度一致
        if len(hist) > self.n_bins:
            hist = hist[:self.n_bins]
        elif len(hist) < self.n_bins:
            hist = np.pad(hist, (0, self.n_bins - len(hist)), 'constant')
        
        return hist

class ColorFeatureExtractor(FeatureExtractor):
    """颜色特征提取器"""
    def __init__(self, bins=32):
        super().__init__()
        self.bins = bins
    
    def extract(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        
        # 提取颜色直方图
        hist_b = cv2.calcHist([image], [0], None, [self.bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [self.bins], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [self.bins], [0, 256]).flatten()
        
        # 归一化直方图
        if np.sum(hist_b) > 0:
            hist_b = hist_b / np.sum(hist_b)
        if np.sum(hist_g) > 0:
            hist_g = hist_g / np.sum(hist_g)
        if np.sum(hist_r) > 0:
            hist_r = hist_r / np.sum(hist_r)
        
        # 计算颜色矩
        color_moments = []
        for i, channel in enumerate(cv2.split(image)):
            # 一阶矩 (平均值)
            moment1 = np.mean(channel)
            # 二阶矩 (标准差)
            moment2 = np.std(channel)
            # 三阶矩 (偏斜度)
            moment3 = np.cbrt(np.mean(np.power(channel - moment1, 3)))
            color_moments.extend([moment1, moment2, moment3])
        
        # 合并所有颜色特征
        color_features = np.concatenate([hist_b, hist_g, hist_r, color_moments])
        
        return color_features

class CNNFeatureExtractor(FeatureExtractor):
    """CNN特征提取器"""
    def __init__(self, model_name='resnet18'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # 移除最后的全连接层
        elif model_name == 'vgg16':
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.model = self.model.features  # 只保留特征提取部分
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, image_path):
        # 读取并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(image)
        
        # 将特征转换为一维向量
        features = features.squeeze().cpu().numpy()
        
        return features

class FeatureFusion:
    """特征融合类"""
    def __init__(self, extractors):
        """
        初始化特征融合器
        
        参数:
        extractors: 特征提取器列表，每个元素是(提取器, 权重)元组
        """
        self.extractors = extractors
    
    def extract(self, image_path):
        """提取融合特征"""
        all_features = []
        for extractor, weight in self.extractors:
            features = extractor.extract(image_path)
            all_features.append(features * weight)
        
        return np.concatenate(all_features)
    
    def extract_batch(self, image_paths):
        """批量提取融合特征"""
        all_features = []
        for path in tqdm(image_paths, desc="提取融合特征"):
            features = self.extract(path)
            all_features.append(features)
        return np.array(all_features) 
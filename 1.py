import numpy as np
import matplotlib.pyplot as plt
import cv2

# 클래스 이름 정의
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 로드 및 분할
per_class_train = 1000
per_class_valid = 1000
per_class_test = 200

x_train = np.load('/Applications/Python 3.12/data/fashion_mnist/x_train.npy')
y_train = np.load('/Applications/Python 3.12/data/fashion_mnist/y_train.npy')
x_test = np.load('/Applications/Python 3.12/data/fashion_mnist/x_test.npy')
y_test = np.load('/Applications/Python 3.12/data/fashion_mnist/y_test.npy')

x_train_set, y_train_set = [], []
x_valid_set, y_valid_set = [], []
x_test_set, y_test_set = [], []

# 각 클래스별로 훈련, 검증, 테스트 데이터 분할
for i in range(10):
    indices = np.where(y_train == i)[0][:per_class_train + per_class_valid]
    x_train_set.extend(x_train[indices[:per_class_train]])
    y_train_set.extend(y_train[indices[:per_class_train]])
    x_valid_set.extend(x_train[indices[per_class_train:per_class_train + per_class_valid]])
    y_valid_set.extend(y_train[indices[per_class_train:per_class_train + per_class_valid]])

for i in range(10):
    indices = np.where(y_test == i)[0][:per_class_test]
    x_test_set.extend(x_test[indices])
    y_test_set.extend(y_test[indices])

x_train_set = np.array(x_train_set)
y_train_set = np.array(y_train_set)
x_valid_set = np.array(x_valid_set)
y_valid_set = np.array(y_valid_set)
x_test_set = np.array(x_test_set)
y_test_set = np.array(y_test_set)

print(f"x_train_set shape: {x_train_set.shape}, y_train_set shape: {y_train_set.shape}")
print(f"x_valid_set shape: {x_valid_set.shape}, y_valid_set shape: {y_valid_set.shape}")
print(f"x_test_set shape: {x_test_set.shape}, y_test_set shape: {y_test_set.shape}")

# 데이터 전처리 및 특징점 추출

# 1. 정규화 (Normalization)
x_train_set = x_train_set / 255.0
x_valid_set = x_valid_set / 255.0
x_test_set = x_test_set / 255.0

# 2. OpenCV를 사용한 가우시안 블러로 노이즈 제거 (Noise Reduction)
x_train_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_train_set])
x_valid_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_valid_set])
x_test_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_test_set])

# 3. 각 이미지에 외곽선 추가 (Adding edges to each image)
def add_edges(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 80, 120)
    edge_image = np.zeros_like(image)
    edge_image[edges > 0] = 1
    return edge_image

x_train_edges = np.array([add_edges(image) for image in x_train_set])
x_valid_edges = np.array([add_edges(image) for image in x_valid_set])
x_test_edges = np.array([add_edges(image) for image in x_test_set])

# 4. 3차원 데이터를 2차원으로 변환 (Reshaping)
x_train_reshaped = x_train_edges.reshape(x_train_edges.shape[0], -1)
x_valid_reshaped = x_valid_edges.reshape(x_valid_edges.shape[0], -1)
x_test_reshaped = x_test_edges.reshape(x_test_edges.shape[0], -1)

# 5. 데이터 표준화 (Standardization)
# 평균과 표준편차를 구함
train_mean = np.mean(x_train_reshaped, axis=0)
train_std = np.std(x_train_reshaped, axis=0)

# 표준화를 수행
x_train_scaled = (x_train_reshaped - train_mean) / train_std
x_valid_scaled = (x_valid_reshaped - train_mean) / train_std
x_test_scaled = (x_test_reshaped - train_mean) / train_std

# 6. 원-핫 인코딩 (One-Hot Encoding)
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train_encoded = one_hot_encode(y_train_set, 10)
y_valid_encoded = one_hot_encode(y_valid_set, 10)
y_test_encoded = one_hot_encode(y_test_set, 10)

print("y_train_encoded shape:", y_train_encoded.shape)
print("y_valid_encoded shape:", y_valid_encoded.shape)
print("y_test_encoded shape:", y_test_encoded.shape)

# 전처리된 데이터에서 각 클래스별로 10개씩 이미지 시각화
plt.figure(figsize=(15, 9))
for i in range(10):
    indices = np.where(y_train_set == i)[0][:10]
    for j, idx in enumerate(indices):
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.imshow(x_train_edges[idx], cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
plt.tight_layout()
plt.show()

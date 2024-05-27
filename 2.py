import numpy as np
import matplotlib.pyplot as plt
import cv2

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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

# 데이터 무작위 섞기
shuffle_indices = np.random.permutation(len(x_train_set))
x_train_set = x_train_set[shuffle_indices]
y_train_set = y_train_set[shuffle_indices]

shuffle_indices = np.random.permutation(len(x_valid_set))
x_valid_set = x_valid_set[shuffle_indices]
y_valid_set = y_valid_set[shuffle_indices]

shuffle_indices = np.random.permutation(len(x_test_set))
x_test_set = x_test_set[shuffle_indices]
y_test_set = y_test_set[shuffle_indices]

print(f"x_train_set shape: {x_train_set.shape}, y_train_set shape: {y_train_set.shape}")
print(f"x_valid_set shape: {x_valid_set.shape}, y_valid_set shape: {y_valid_set.shape}")
print(f"x_test_set shape: {x_test_set.shape}, y_test_set shape: {y_test_set.shape}")

# 1. 정규화 (Normalization)
x_train_set = x_train_set / 255.0
x_valid_set = x_valid_set / 255.0
x_test_set = x_test_set / 255.0

# 2. 가우시안 블러로 노이즈 제거
x_train_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_train_set])
x_valid_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_valid_set])
x_test_set = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_test_set])

# 3. 3차원 데이터를 1차원으로 변환
x_train_reshaped = x_train_set.reshape(x_train_set.shape[0], -1)
x_valid_reshaped = x_valid_set.reshape(x_valid_set.shape[0], -1)
x_test_reshaped = x_test_set.reshape(x_test_set.shape[0], -1)

# 4. 데이터 표준화 (Standardization)
# 평균과 표준편차를 구함
train_mean = np.mean(x_train_reshaped, axis=0)
train_std = np.std(x_train_reshaped, axis=0)
    
# 표준화를 수행
x_train_scaled = (x_train_reshaped - train_mean) / train_std
x_valid_scaled = (x_valid_reshaped - train_mean) / train_std
x_test_scaled = (x_test_reshaped - train_mean) / train_std

# 원-핫 인코딩 (One-Hot Encoding)
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train_encoded = one_hot_encode(y_train_set, 10)
y_valid_encoded = one_hot_encode(y_valid_set, 10)
y_test_encoded = one_hot_encode(y_test_set, 10)

print("y_train_encoded shape:", y_train_encoded.shape)
print("y_valid_encoded shape:", y_valid_encoded.shape)
print("y_test_encoded shape:", y_test_encoded.shape)

# 전처리된 데이터에서 각 클래스별로 모든 이미지 시각화
def plot_images(images, labels, class_names, start_index, num_images):
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(num_images): 
        img = images[start_index + i].reshape(28, 28)

        # 윤곽선을 검출하여 이미지에 그리기
        contours, _ = cv2.findContours((img * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 1)

        ax = axes[i]
        ax.imshow(img_contour)
        ax.set_title(class_names[labels[start_index + i]])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 페이지 당 100개의 이미지를 표시
num_images_per_page = 100
total_images = x_train_set.shape[0]
num_pages = (total_images // num_images_per_page) + 1

for page in range(num_pages):
    start_index = page * num_images_per_page
    end_index = min(start_index + num_images_per_page, total_images)
    plot_images(x_train_set, y_train_set, class_names, start_index, end_index - start_index)
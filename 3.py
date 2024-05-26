import numpy as np
import matplotlib.pyplot as plt
import cv2

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
per_class_train = 1000
per_class_valid = 1000
per_class_test = 200
n = 7  # 나눌 구역의 크기

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

# 윤곽선 검출 및 구역별 평균 픽셀 값 계산
def calculate_average_pixel_values(images, labels, class_idx, n):
    class_images = images[labels == class_idx]
    avg_pixel_values = np.zeros((n, n))

    for img in class_images:
        h, w = img.shape
        step_h, step_w = h // n, w // n
        
        # 윤곽선 검출
        contours, _ = cv2.findContours((img * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 1)

        for i in range(n):
            for j in range(n):
                start_h, end_h = i * step_h, (i + 1) * step_h
                start_w, end_w = j * step_w, (j + 1) * step_w
                region = img[start_h:end_h, start_w:end_w]
                
                # 테두리 안쪽의 픽셀 값만 계산
                mask = np.zeros(region.shape, dtype=np.uint8)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
                masked_region = cv2.bitwise_and(region, region, mask=mask[start_h:end_h, start_w:end_w])
                
                avg_pixel_values[i, j] += np.mean(masked_region)
    
    avg_pixel_values /= len(class_images)
    return avg_pixel_values

# 각 클래스별 추상화된 이미지 생성 및 시각화
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for class_idx, class_name in enumerate(class_names):
    avg_pixel_values = calculate_average_pixel_values(x_train_set, y_train_set, class_idx, n)
    
    # 추상화된 이미지 생성
    abstract_image = np.zeros((28, 28))
    step_h, step_w = 28 // n, 28 // n

    for i in range(n):
        for j in range(n):
            start_h, end_h = i * step_h, (i + 1) * step_h
            start_w, end_w = j * step_w, (j + 1) * step_w
            abstract_image[start_h:end_h, start_w:end_w] = avg_pixel_values[i, j]
    
    axes[class_idx].imshow(abstract_image, cmap='gray')
    axes[class_idx].set_title(class_name)
    axes[class_idx].axis('off')

plt.tight_layout()
plt.show()

# 각 클래스별 평균 픽셀 값을 계산하고 시각화 (막대 그래프)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for class_idx, class_name in enumerate(class_names):
    avg_pixel_values = calculate_average_pixel_values(x_train_set, y_train_set, class_idx, n)
    avg_pixel_values_flat = avg_pixel_values.flatten()
    
    ax = axes[class_idx]
    ax.bar(range(len(avg_pixel_values_flat)), avg_pixel_values_flat)
    ax.set_title(class_name)
    ax.set_xlabel('area')
    ax.set_ylabel('Average Pixel Value')

plt.tight_layout()
plt.show()

# 각 클래스별 평균 픽셀 값을 계산하고 시각화 (꺾은선 그래프)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for class_idx, class_name in enumerate(class_names):
    avg_pixel_values = calculate_average_pixel_values(x_train_set, y_train_set, class_idx, n)
    avg_pixel_values_flat = avg_pixel_values.flatten()
    
    ax = axes[class_idx]
    ax.plot(range(len(avg_pixel_values_flat)), avg_pixel_values_flat, marker='o')
    ax.set_title(class_name)
    ax.set_xlabel('Region')
    ax.set_ylabel('Average Pixel Value')

plt.tight_layout()
plt.show()

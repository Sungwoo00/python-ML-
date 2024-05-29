import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 클래스 이름과 데이터 개수 설정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
per_class_train = 1000
per_class_valid = 1000
per_class_test = 200
n = 7

# 데이터 로드
x_train = np.load('/Applications/Python 3.12/data/fashion_mnist/x_train.npy')
y_train = np.load('/Applications/Python 3.12/data/fashion_mnist/y_train.npy')
x_test = np.load('/Applications/Python 3.12/data/fashion_mnist/x_test.npy')
y_test = np.load('/Applications/Python 3.12/data/fashion_mnist/y_test.npy')

# 데이터 분할 함수
def split_data(x_data, y_data, per_class_train, per_class_valid):
    x_train_set, y_train_set = [], []
    x_valid_set, y_valid_set = [], []
    for i in range(10):
        indices = np.where(y_data == i)[0][:per_class_train + per_class_valid]
        x_train_set.extend(x_data[indices[:per_class_train]])
        y_train_set.extend(y_data[indices[:per_class_train]])
        if per_class_valid > 0:
            x_valid_set.extend(x_data[indices[per_class_train:per_class_train + per_class_valid]])
            y_valid_set.extend(y_data[indices[per_class_train:per_class_train + per_class_valid]])
    return np.array(x_train_set), np.array(y_train_set), np.array(x_valid_set), np.array(y_valid_set)

# 데이터 분할
x_train_set, y_train_set, x_valid_set, y_valid_set = split_data(x_train, y_train, per_class_train, per_class_valid)
x_test_set, y_test_set, _, _ = split_data(x_test, y_test, per_class_test, 0)

# 데이터 무작위 섞기
def shuffle_data(x_data, y_data):
    shuffle_indices = np.random.permutation(len(x_data))
    return x_data[shuffle_indices], y_data[shuffle_indices]

x_train_set, y_train_set = shuffle_data(x_train_set, y_train_set)
x_valid_set, y_valid_set = shuffle_data(x_valid_set, y_valid_set)
x_test_set, y_test_set = shuffle_data(x_test_set, y_test_set)

print(f"x_train_set shape: {x_train_set.shape}, y_train_set shape: {y_train_set.shape}")
print(f"x_valid_set shape: {x_valid_set.shape}, y_valid_set shape: {y_valid_set.shape}")
print(f"x_test_set shape: {x_test_set.shape}, y_test_set shape: {y_test_set.shape}")

# 1. 정규화
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

# 윤곽선 검출 및 구역별 평균 픽셀 값 계산 함수
def calculate_average_pixel_values(images, labels, class_idx, n):
    class_images = images[labels == class_idx]
    avg_pixel_values = np.zeros((n, n))

    for img in class_images:
        h, w = img.shape
        step_h, step_w = h // n, w // n
        
        contours, _ = cv2.findContours((img * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 1)

        for i in range(n):
            for j in range(n):
                start_h, end_h = i * step_h, (i + 1) * step_h
                start_w, end_w = j * step_w, (j + 1) * step_w
                region = img[start_h:end_h, start_w:end_w]
                
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

# 각 클래스별 평균 픽셀 값 계산 및 시각화 (막대 그래프)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for class_idx, class_name in enumerate(class_names):
    avg_pixel_values = calculate_average_pixel_values(x_train_set, y_train_set, class_idx, n)
    avg_pixel_values_flat = avg_pixel_values.flatten()
    
    ax = axes[class_idx]
    ax.bar(range(len(avg_pixel_values_flat)), avg_pixel_values_flat)
    ax.set_title(class_name)
    ax.set_xlabel('Area')
    ax.set_ylabel('Average Pixel Value')

plt.tight_layout()
plt.show()

# 각 클래스별 평균 픽셀 값 계산 및 시각화 (꺾은선 그래프)
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

# 인식기 적용 및 성능 평가
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,))
}

for name, clf in classifiers.items():
    # train 데이터로 학습하고 valid 데이터로 성능을 평가
    clf.fit(x_train_scaled, y_train_set)
    y_valid_pred = clf.predict(x_valid_scaled)
    valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
    valid_confusion_matrix = confusion_matrix(y_valid_set, y_valid_pred)
    print(f"{name} Valid Accuracy: {valid_accuracy}")
    print(f"{name} Valid Confusion Matrix:\n{valid_confusion_matrix}")

    # test 데이터로 최종 성능을 평가
    y_test_pred = clf.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    test_confusion_matrix = confusion_matrix(y_test_set, y_test_pred)
    print(f"{name} Test Accuracy: {test_accuracy}")
    print(f"{name} Test Confusion Matrix:\n{test_confusion_matrix}")

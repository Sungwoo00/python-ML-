import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 클래스 이름과 데이터 개수 설정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
per_class_train = 1000
per_class_valid = 1000
per_class_test = 200
n = 14

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

print(f"x_train_set shape: {x_train_set.shape}, y_train_set shape: {y_train_set.shape}")
print(f"x_valid_set shape: {x_valid_set.shape}, y_valid_set shape: {y_valid_set.shape}")
print(f"x_test_set shape: {x_test_set.shape}, y_test_set shape: {y_test_set.shape}")

# 전처리 단계별 정확도 기록
accuracy_results = []

def record_accuracy(stage, x_train_processed, x_valid_processed):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }
    
    stage_accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(x_train_processed, y_train_set)
        y_valid_pred = clf.predict(x_valid_processed)
        valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
        stage_accuracies[name] = valid_accuracy
    accuracy_results.append((stage, stage_accuracies))

# 1. 정규화
x_train_normalized = x_train_set / 255.0
x_valid_normalized = x_valid_set / 255.0
record_accuracy("Normalization", x_train_normalized.reshape(x_train_normalized.shape[0], -1), x_valid_normalized.reshape(x_valid_normalized.shape[0], -1))

# 2. 가우시안 블러로 노이즈 제거
x_train_blurred = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_train_normalized])
x_valid_blurred = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_valid_normalized])
record_accuracy("Gaussian Blur", x_train_blurred.reshape(x_train_blurred.shape[0], -1), x_valid_blurred.reshape(x_valid_blurred.shape[0], -1))

# 3. 3차원 데이터를 1차원으로 변환 (이미 1번, 2번에서 수행됨)

# 4. 데이터 표준화 (Standardization)
train_mean = np.mean(x_train_blurred.reshape(x_train_blurred.shape[0], -1), axis=0)
train_std = np.std(x_train_blurred.reshape(x_train_blurred.shape[0], -1), axis=0)
x_train_standardized = (x_train_blurred.reshape(x_train_blurred.shape[0], -1) - train_mean) / train_std
x_valid_standardized = (x_valid_blurred.reshape(x_valid_blurred.shape[0], -1) - train_mean) / train_std
record_accuracy("Standardization", x_train_standardized, x_valid_standardized)

# 5. 윤곽선 검출 및 구역별 평균 픽셀 값 계산 함수
def calculate_average_pixel_values(images, n):
    avg_pixel_values_set = []
    for img in images:
        h, w = img.shape
        step_h, step_w = h // n, w // n
        
        contours, _ = cv2.findContours((img * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        avg_pixel_values = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                start_h, end_h = i * step_h, (i + 1) * step_h
                start_w, end_w = j * step_w, (j + 1) * step_w
                region = img[start_h:end_h, start_w:end_w]
                
                mask = np.zeros(region.shape, dtype=np.uint8)
                cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
                masked_region = cv2.bitwise_and(region, region, mask=mask)
                
                avg_pixel_values[i, j] = np.mean(masked_region)
        
        avg_pixel_values_set.append(avg_pixel_values.flatten())
    return np.array(avg_pixel_values_set)

# 윤곽선 검출 및 구역별 평균 픽셀 값 계산 적용
x_train_contour = calculate_average_pixel_values(x_train_standardized.reshape(-1, 28, 28), n)
x_valid_contour = calculate_average_pixel_values(x_valid_standardized.reshape(-1, 28, 28), n)
record_accuracy("Contour and Pixel Average", x_train_contour, x_valid_contour)

# 정확도 그래프 출력
stages = [result[0] for result in accuracy_results]
logistic_accuracies = [result[1]["Logistic Regression"] for result in accuracy_results]
decision_tree_accuracies = [result[1]["Decision Tree"] for result in accuracy_results]
mlp_accuracies = [result[1]["MLP"] for result in accuracy_results]

plt.figure(figsize=(12, 8))
plt.plot(stages, logistic_accuracies, marker='o', label='Logistic Regression')
plt.plot(stages, decision_tree_accuracies, marker='o', label='Decision Tree')
plt.plot(stages, mlp_accuracies, marker='o', label='MLP')
plt.xlabel('Preprocessing Stage')
plt.ylabel('Accuracy')
plt.title('Accuracy by Preprocessing Stage')
plt.legend()
plt.grid(True)
plt.show()

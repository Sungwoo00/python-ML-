########################################################################################################################################################################################################
# 전처리 전 데이터
########################################################################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 클래스 이름과 데이터 개수 설정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
per_class_train = 1000
per_class_valid = 1000
per_class_test = 200

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

x_train_raw = x_train_set
x_valid_raw = x_valid_set
x_test_raw = x_test_set

# 인식기별 파라미터 설정 
def get_classifier_and_params():
    classifiers = {
        'Logistic Regression': (
            LogisticRegression, 
            np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1), np.arange(1, 10, 1),
                            np.arange(10, 100, 10), np.arange(100, 1000 + 100, 100)))
        ),
        'Decision Tree': (
            DecisionTreeClassifier, 
            np.arange(1, 51, 1)
        ),
        'MLP': (
            MLPClassifier, 
            [tuple([n]) for n in np.concatenate((np.arange(10, 100, 10), np.arange(100, 3100 , 100)))]
        )
    }
    return classifiers

# 파라미터 변화율별 정확도 기록
def record_parameter_accuracy(classifier, param_name, param_values, x_train, y_train, x_valid, y_valid):
    train_accuracies = []
    valid_accuracies = []
    for value in param_values:
        if param_name in ['C', 'maㅌx_depth', 'hidden_layer_sizes']:
            clf = classifier(**{param_name: value, 'max_iter': 1000}) if param_name != 'max_depth' else classifier(**{param_name: value})
        else:
            clf = classifier(**{param_name: value})
        
        clf.fit(x_train, y_train)
        
        # Train 정확도 계산
        y_train_pred = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        # Validation 정확도 계산
        y_valid_pred = clf.predict(x_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_accuracies.append(valid_accuracy)

        print(f"{param_name} = {value}: Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {valid_accuracy:.4f}")
    
    # 최적 파라미터 찾기
    optimal_index = None
    max_valid_accuracy = -1
    declining = False
    initial_decline_index = None

    for i in range(1, len(valid_accuracies) - 1):
        if valid_accuracies[i] < valid_accuracies[i - 1]:  
            if not declining:
                declining = True
                initial_decline_index = i - 1
                max_valid_accuracy = valid_accuracies[initial_decline_index]
            elif valid_accuracies[i] < max_valid_accuracy and train_accuracies[i] > train_accuracies[i - 1]:
                optimal_index = initial_decline_index
        elif declining and valid_accuracies[i] > max_valid_accuracy:
            max_valid_accuracy = valid_accuracies[i]
            initial_decline_index = i

    if optimal_index is None:
        print("No optimal parameter found.")
        optimal_param = None
    else:
        optimal_param = param_values[optimal_index]
    
    return train_accuracies, valid_accuracies, optimal_param

# 분류기와 파라미터 정보 가져오기
classifiers_params = get_classifier_and_params()

optimal_params = {}

plt.figure(figsize=(15, 5))

for idx, (name, (Classifier, params)) in enumerate(classifiers_params.items(), 1):
    # 데이터셋 설정
    x_train = x_train_raw.reshape(x_train_raw.shape[0], -1)
    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], -1)
    y_train = y_train_set
    y_valid = y_valid_set
    
    # 정확도 기록 및 최적 파라미터 찾기
    param_name = 'hidden_layer_sizes' if name == 'MLP' else 'C' if name == 'Logistic Regression' else 'max_depth'
    train_accuracies, valid_accuracies, optimal_param = record_parameter_accuracy(Classifier, param_name, params, x_train, y_train, x_valid, y_valid)
    optimal_params[name] = optimal_param

    # 그래프 출력
    plt.subplot(1, len(classifiers_params), idx)
    plt.plot(params, train_accuracies, marker='o', markersize=2, linestyle='-', color='r', label='Train')
    plt.plot(params, valid_accuracies, marker='o', markersize=2, linestyle='-', color='b', label='Validation')
    plt.axvline(x=optimal_param, color='orange', linestyle='--', label=f'Optimal {optimal_param}')
    plt.xlabel('C value' if name == 'Logistic Regression' else 'Max Depth' if name == 'Decision Tree' else 'Hidden Layer Sizes')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.grid(True)
    plt.legend()

    print(f"{name} 최적 파라미터 값: {optimal_param}")

plt.tight_layout()
plt.show()

# 최적 파라미터로 최종 성능 평가
final_accuracies_test = {}
# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# 분류 성능 지표 시각화 함수
def plot_classification_report(y_true, y_pred, class_names, title):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
    plt.title(title)
    plt.show()

for name, optimal_param in optimal_params.items():
    Classifier, _ = classifiers_params[name]
    if name == 'MLP':
        clf = Classifier(hidden_layer_sizes=optimal_param)
    elif name == 'Logistic Regression':
        clf = Classifier(C=optimal_param, max_iter=1000)
    elif name == 'Decision Tree':
        clf = Classifier(max_depth=optimal_param)
    else:
        clf = Classifier()
    clf.fit(x_train_raw.reshape(x_train_raw.shape[0], -1), y_train_set)
    
    # Test 데이터 기준 혼동 행렬 및 분류 성능 지펴ㅛ
    y_test_pred = clf.predict(x_test_raw.reshape(x_test_raw.shape[0], -1))
    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    final_accuracies_test[name] = test_accuracy

    print(f"\n{name} - Test Data Classification Report:")
    print(classification_report(y_test_set, y_test_pred, target_names=class_names))
    
    plot_confusion_matrix(y_test_set, y_test_pred, class_names, f'{name} - Test Data')
    plot_classification_report(y_test_set, y_test_pred, class_names, f'{name} - Test Data')

# 최종 성능 결과 출력
print("최적 파라미터 기반 최종 성능 결과:")
print("Test 데이터:")
for name, accuracy in final_accuracies_test.items():
    print(f"{name}: {accuracy:.4f}")

# 최종 성능 결과 그래프
plt.figure(figsize=(10, 5))
names = list(final_accuracies_test.keys())
test_scores = list(final_accuracies_test.values())

bar_width = 0.35
index = np.arange(len(names))

plt.bar(index, test_scores, bar_width, label='Test')

plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Performance with Optimal Parameters')
plt.xticks(index, names)
plt.legend()

plt.tight_layout()
plt.show()
########################################################################################################################################################################################################
# 전처리 후
########################################################################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
accuracy_results_train = []
accuracy_results_valid = []
accuracy_results_test = []
steps = []

def record_accuracy(step, x_train_processed, x_test_processed):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    
    test_accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(x_train_processed, y_train_set)
        
        y_test_pred = clf.predict(x_test_processed)
        test_accuracy = accuracy_score(y_test_set, y_test_pred)
        test_accuracies[name] = test_accuracy
        
    accuracy_results_test.append((step, test_accuracies))
    steps.append(step)
    
# 0. 원본 데이터 (정규화 전)
x_train_raw = x_train_set
x_valid_raw = x_valid_set
x_test_raw = x_test_set
record_accuracy("Raw", x_train_raw.reshape(x_train_raw.shape[0], -1), x_valid_raw.reshape(x_valid_raw.shape[0], -1), x_test_raw.reshape(x_test_raw.shape[0], -1))

# 1. 정규화
x_train_normalized = x_train_set / 255.0
x_valid_normalized = x_valid_set / 255.0
x_test_normalized = x_test_set / 255.0
record_accuracy("Normalization", x_train_normalized.reshape(x_train_normalized.shape[0], -1), x_valid_normalized.reshape(x_valid_normalized.shape[0], -1), x_test_normalized.reshape(x_test_normalized.shape[0], -1))

# 2. 가우시안 블러로 노이즈 제거
x_train_blurred = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_train_normalized])
x_valid_blurred = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_valid_normalized])
x_test_blurred = np.array([cv2.GaussianBlur(image, (3, 3), 0) for image in x_test_normalized])
record_accuracy("Gaussian Blur", x_train_blurred.reshape(x_train_blurred.shape[0], -1), x_valid_blurred.reshape(x_valid_blurred.shape[0], -1), x_test_blurred.reshape(x_test_blurred.shape[0], -1))

# 3. 데이터 표준화 (Standardization)
train_mean = np.mean(x_train_blurred.reshape(x_train_blurred.shape[0], -1), axis=0)
train_std = np.std(x_train_blurred.reshape(x_train_blurred.shape[0], -1), axis=0)
x_train_standardized = (x_train_blurred.reshape(x_train_blurred.shape[0], -1) - train_mean) / train_std
x_valid_standardized = (x_valid_blurred.reshape(x_valid_blurred.shape[0], -1) - train_mean) / train_std
x_test_standardized = (x_test_blurred.reshape(x_test_blurred.shape[0], -1) - train_mean) / train_std
record_accuracy("Standardization", x_train_standardized, x_valid_standardized, x_test_standardized)

# 4. 윤곽선 검출 및 구역별 평균 픽셀 값 계산 함수
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
x_test_contour = calculate_average_pixel_values(x_test_standardized.reshape(-1, 28, 28), n)
record_accuracy("Contour and Pixel Average", x_train_contour, x_valid_contour, x_test_contour)

# 전처리 단계별 정확도 그래프 출력
steps, train_accuracies = zip(*accuracy_results_train)
plt.figure(figsize=(10, 6))
plt.plot(steps, train_accuracies, marker='o', label='Train Accuracy', linestyle='-')
plt.xlabel('Preprocessing step')
plt.ylabel('Accuracy')
plt.title('Logistic Regression - Accuracy at each preprocessing step')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 인식기별 파라미터 설정 
def get_classifier_and_params():
    classifiers = {
        'Logistic Regression': (
            LogisticRegression, 
            np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1), 
                            np.arange(10, 100, 10), np.arange(100, 1000 + 100, 100)))
        ),
        'Decision Tree': (
            DecisionTreeClassifier, 
            np.arange(1, 50 + 1, 1)
        ),
        'MLP': (
            MLPClassifier, 
            [tuple([n]) for n in np.concatenate((np.arange(10, 100, 10), np.arange(100, 3100 , 100)))]
        )
    }
    return classifiers

# 파라미터 변화율별 정확도 기록
def record_parameter_accuracy(classifier, param_name, param_values, x_train, y_train, x_valid, y_valid):
    valid_accuracies = []
    train_accuracies = []
    best_valid_accuracy = -np.inf
    best_valid_index = -1
    drop_detect = 0
    for idx, value in enumerate(param_values):
        if param_name in ['C', 'max_depth', 'hidden_layer_sizes']:
            clf = classifier(**{param_name: value, 'max_iter': 1000}) if param_name != 'max_depth' else classifier(**{param_name: value})
        else:
            clf = classifier(**{param_name: value})
        
        clf.fit(x_train, y_train)
        
        # Train 정확도
        y_train_pred = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        # Validation 정확도
        y_valid_pred = clf.predict(x_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_accuracies.append(valid_accuracy)
        
        print(f"{param_name} = {value}: Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {valid_accuracy:.4f}")
        
        # 연속적인 정확도 저하 체크 
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_valid_index = idx
            drop_detect = 0
        else:
            drop_detect += 1

    optimal_param = param_values[best_valid_index]
    return train_accuracies, valid_accuracies, optimal_param

classifiers_params = get_classifier_and_params()

optimal_params = {}

plt.figure(figsize=(15, 5))

for idx, (name, (Classifier, params)) in enumerate(classifiers_params.items(), 1):
    # 정확도 기록
    x_train = x_train_contour
    x_valid = x_valid_contour
    y_train = y_train_set
    y_valid = y_valid_set
    
    train_accuracies, valid_accuracies, optimal_param = record_parameter_accuracy(Classifier, 'hidden_layer_sizes' if name == 'MLP' else 'C' if name == 'Logistic Regression' else 'max_depth', params, x_train, y_train, x_valid, y_valid)
    optimal_params[name] = optimal_param

    # 그래프 출력
    plt.subplot(1, 3, idx)
    plt.plot(params[:len(train_accuracies)], train_accuracies, marker='o', linestyle='-', color='r', label='Train')
    plt.plot(params[:len(valid_accuracies)], valid_accuracies, marker='o', linestyle='-', color='b', label='Validation')
    plt.axvline(x=optimal_param, color='orange', linestyle='--')
    plt.xlabel('C value' if name == 'Logistic Regression' else 'Max Depth' if name == 'Decision Tree' else 'Hidden Layer Sizes')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.grid(True)
    plt.legend()

    print(f"{name} 최적 파라미터 값: {optimal_param}")

plt.tight_layout()
plt.show()

# 최적 파라미터로 최종 성능 평가
final_accuracies_valid = {}
final_accuracies_test = {}

for name, optimal_param in optimal_params.items():
    Classifier, _ = classifiers_params[name]
    clf = Classifier(**{'max_iter': optimal_param} if name == 'MLP' else {'C': optimal_param} if name == 'Logistic Regression' else {'max_depth': optimal_param})
    clf.fit(x_train_contour, y_train_set)
    
    y_valid_pred = clf.predict(x_valid_contour)
    valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
    final_accuracies_valid[name] = valid_accuracy
    
    y_test_pred = clf.predict(x_test_contour)
    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    final_accuracies_test[name] = test_accuracy

# 최종 성능 결과 출력
print("최적 파라미터 기반 최종 성능 결과:")
# print("Validation 데이터:")
# for name, accuracy in final_accuracies_valid.items():
#     print(f"{name}: {accuracy:.4f}")
print("Test 데이터:")
for name, accuracy in final_accuracies_test.items():
    print(f"{name}: {accuracy:.4f}")

# 최종 성능 결과 그래프
plt.figure(figsize=(10, 5))
names = list(final_accuracies_valid.keys())
valid_scores = list(final_accuracies_valid.values())
test_scores = list(final_accuracies_test.values())

bar_width = 0.35
index = np.arange(len(names))

plt.bar(index, valid_scores, bar_width, label='Validation')
plt.bar(index + bar_width, test_scores, bar_width, label='Test')

plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Performance with Optimal Parameters')
plt.xticks(index + bar_width / 2, names)
plt.legend()

plt.tight_layout()
plt.show()


# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# 최적 파라미터로 최종 성능 평가 및 혼동 행렬 시각화
for name, optimal_param in optimal_params.items():
    Classifier, _ = classifiers_params[name]
    clf = Classifier(**{'max_iter': optimal_param} if name == 'MLP' else {'C': optimal_param} if name == 'Logistic Regression' else {'max_depth': optimal_param})
    clf.fit(x_train_contour, y_train_set)
    
    # # Validation 혼동 행렬
    # y_valid_pred = clf.predict(x_valid_contour)
    # plot_confusion_matrix(y_valid_set, y_valid_pred, class_names, f'{name} - Validation Data')
    
    # Test 혼동 행렬
    y_test_pred = clf.predict(x_test_contour)
    plot_confusion_matrix(y_test_set, y_test_pred, class_names, f'{name} - Test Data')

    # valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
    # final_accuracies_valid[name] = valid_accuracy   

    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    final_accuracies_test[name] = test_accuracy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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

# 원본 데이터 (정규화 전)
x_train_raw = x_train_set
x_valid_raw = x_valid_set
x_test_raw = x_test_set

x_train_raw = x_train_raw.reshape(x_train_raw.shape[0], -1)
x_valid_raw = x_valid_raw.reshape(x_valid_raw.shape[0], -1)
x_test_raw = x_test_raw.reshape(x_test_raw.shape[0], -1)

# 인식기별 파라미터 설정 
def get_classifier_and_params():
    classifiers = {
        # 'Logistic Regression': (
        #     LogisticRegression, 
        #     np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1, 0.1), np.arange(1, 10, 1), 
        #                     np.arange(10, 100, 10), np.arange(100, 1000 + 100, 100)))
        # ),
        # 'Decision Tree': (
        #     DecisionTreeClassifier, 
        #     np.arange(5, 50 + 1, 5)
        # ),
        'MLP': (
            MLPClassifier, 
            [tuple([n]) for n in np.concatenate((np.arange(10, 100, 10), np.arange(100, 3000 + 100, 100)))]
        )
    }
    return classifiers

# 파라미터 변화율별 정확도 기록
def record_parameter_accuracy(classifier, param_name, param_values, x_train, y_train, x_valid, y_valid):
    valid_accuracies = []
    train_accuracies = []
    best_valid_accuracy = -np.inf
    best_valid_index = -1
    consecutive_drops = 0
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
        
        # 연속적인 정확도 저하 체크 (break 사용 안함)
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_valid_index = idx
            consecutive_drops = 0
        else:
            consecutive_drops += 1

    optimal_param = param_values[best_valid_index]
    return train_accuracies, valid_accuracies, optimal_param

classifiers_params = get_classifier_and_params()

optimal_params = {}

plt.figure(figsize=(15, 5))

for idx, (name, (Classifier, params)) in enumerate(classifiers_params.items(), 1):
    # 정확도 기록
    x_train = x_train_raw
    x_valid = x_valid_raw
    y_train = y_train_set
    y_valid = y_valid_set
    
    train_accuracies, valid_accuracies, optimal_param = record_parameter_accuracy(Classifier, 'hidden_layer_sizes' if name == 'MLP' else 'C' if name == 'Logistic Regression' else 'max_depth', params, x_train, y_train, x_valid, y_valid)
    optimal_params[name] = optimal_param

    # 그래프 출력
    plt.subplot(1, 1, idx)
    plt.plot(params[:len(train_accuracies)], train_accuracies, marker='o', linestyle='-', color='r', label='Train')
    plt.plot(params[:len(valid_accuracies)], valid_accuracies, marker='o', linestyle='-', color='b', label='Validation')
    plt.axvline(x=optimal_param, color='orange', linestyle='--')
    plt.xlabel('Hidden Layer Sizes')
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
    clf = Classifier(hidden_layer_sizes=optimal_param, max_iter=1000)
    clf.fit(x_train_raw, y_train_set)
    
    y_valid_pred = clf.predict(x_valid_raw)
    valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
    final_accuracies_valid[name] = valid_accuracy
    
    y_test_pred = clf.predict(x_test_raw)
    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    final_accuracies_test[name] = test_accuracy

# 최종 성능 결과 출력
print("최적 파라미터 기반 최종 성능 결과:")
print("Validation 데이터:")
for name, accuracy in final_accuracies_valid.items():
    print(f"{name}: {accuracy:.4f}")
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
    clf = Classifier(hidden_layer_sizes=optimal_param, max_iter=1000)
    clf.fit(x_train_raw, y_train_set)
    
    # Validation 혼동 행렬
    y_valid_pred = clf.predict(x_valid_raw)
    plot_confusion_matrix(y_valid_set, y_valid_pred, class_names, f'{name} - Validation Data')
    
    # Test 혼동 행렬
    y_test_pred = clf.predict(x_test_raw)
    plot_confusion_matrix(y_test_set, y_test_pred, class_names, f'{name} - Test Data')

    valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
    final_accuracies_valid[name] = valid_accuracy   

    test_accuracy = accuracy_score(y_test_set, y_test_pred)
    final_accuracies_test[name] = test_accuracy

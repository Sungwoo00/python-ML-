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

# # 전처리 단계별 정확도 기록
# accuracy_results_valid = []
# accuracy_results_test = []

# def record_accuracy(step, x_train_processed, x_valid_processed, x_test_processed):
#     classifiers = {
#         "Logistic Regression": LogisticRegression(max_iter=1000),
#         "Decision Tree": DecisionTreeClassifier(max_depth=10),
#         "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
#     }
    
#     valid_accuracies = {}
#     test_accuracies = {}
#     for name, clf in classifiers.items():
#         clf.fit(x_train_processed, y_train_set)
        
#         y_valid_pred = clf.predict(x_valid_processed)
#         valid_accuracy = accuracy_score(y_valid_set, y_valid_pred)
#         valid_accuracies[name] = valid_accuracy
        
#         y_test_pred = clf.predict(x_test_processed)
#         test_accuracy = accuracy_score(y_test_set, y_test_pred)
#         test_accuracies[name] = test_accuracy
        
#     accuracy_results_valid.append((step, valid_accuracies))
#     accuracy_results_test.append((step, test_accuracies))

x_train_raw = x_train_set
x_valid_raw = x_valid_set
x_test_raw = x_test_set
# record_accuracy("Raw", x_train_raw.reshape(x_train_raw.shape[0], -1), x_valid_raw.reshape(x_valid_raw.shape[0], -1), x_test_raw.reshape(x_test_raw.shape[0], -1))

# 인식기별 파라미터 설정 
def get_classifier_and_params():
    classifiers = {
        # 'Logistic Regression': (
        #     LogisticRegression, 
        #     np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1.1, 0.1), 
        #                     np.arange(10, 100, 10), np.arange(100, 1000 + 100, 100)))
        # ),
        # 'Decision Tree': (
        #     DecisionTreeClassifier, 
        #     np.arange(1, 51, 1)
        # ),
        'MLP': (
            MLPClassifier, 
            [tuple([n]) for n in np.concatenate((np.arange(10, 100, 10), np.arange(100, 2100 , 100)))]
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
        
        # Train 정확도
        y_train_pred = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        # Validation 정확도
        y_valid_pred = clf.predict(x_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_accuracies.append(valid_accuracy)

        print(f"{param_name} = {value}: Train Accuracy = {train_accuracy:.4f}, Validation Accuracy = {valid_accuracy:.4f}")
    
    return train_accuracies, valid_accuracies

classifiers_params = get_classifier_and_params()

optimal_params = {}

plt.figure(figsize=(15, 5))

for idx, (name, (Classifier, params)) in enumerate(classifiers_params.items(), 1):
    # 정확도 기록
    x_train = x_train_raw.reshape(x_train_raw.shape[0], -1)
    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], -1)
    y_train = y_train_set
    y_valid = y_valid_set
    
    train_accuracies, valid_accuracies = record_parameter_accuracy(Classifier, 'hidden_layer_sizes' if name == 'MLP' else 'C' if name == 'Logistic Regression' else 'max_depth', params, x_train, y_train, x_valid, y_valid)
    
    # 최적 파라미터 찾기: Train은 상승, Valid는 하락 추세를 보이는 지점
    optimal_param = params[0]
    consecutive_drop_count = 0
    min_diff = float('inf')
    min_diff_idx = 0
    last_single_drop_idx = -1

    for i in range(1, len(params)):
        if train_accuracies[i] > train_accuracies[i - 1] and valid_accuracies[i] < valid_accuracies[i - 1]:
            consecutive_drop_count += 1
            if consecutive_drop_count >= 2:
                optimal_param = params[i - 1]
        else:
            consecutive_drop_count = 0

        # 차이가 가장 적은 지점 추출
        diff = abs(train_accuracies[i] - valid_accuracies[i])
        if diff < min_diff:
            min_diff = diff
            min_diff_idx = i

    # 두 번 연속된 감소가 없으면 최소한 한 번 감소한 지점 사용, 없으면 차이가 가장 적은 지점 사용
    if consecutive_drop_count < 2:
        if last_single_drop_idx != -1:
            optimal_param = params[last_single_drop_idx]
        else:
            optimal_param = params[min_diff_idx]

    optimal_params[name] = optimal_param

    # 그래프 출력
    plt.subplot(1, 3, idx)
    plt.plot(params, train_accuracies, marker='o', markersize=2, linestyle='-', color='r', label='Train')
    plt.plot(params, valid_accuracies, marker='o', markersize=2, linestyle='-', color='b', label='Validation')
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
        clf = Classifier(hidden_layer_sizes=optimal_param, max_iter=1000)
    elif name == 'Logistic Regression':
        clf = Classifier(C=optimal_param, max_iter=1000)
    elif name == 'Decision Tree':
        clf = Classifier(max_depth=optimal_param)
    else:
        clf = Classifier()
    clf.fit(x_train_raw.reshape(x_train_raw.shape[0], -1), y_train_set)
    
    # Test 데이터에 대해 혼동 행렬 및 분류 성능 지표 출력
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
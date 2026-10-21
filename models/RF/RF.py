import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools
import warnings
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# ======================
# 忽略警告
# ======================
warnings.filterwarnings("ignore")

# ======================
# 混淆矩阵可视化函数
# ======================
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, SIZE=6):
    """绘制混淆矩阵"""
    plt.figure(figsize=(SIZE, SIZE))
    
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15, y=1.02)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=50, fontsize=12)
    plt.yticks(tick_marks, classes, rotation=50, fontsize=12)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()

# ======================
# 数据加载与分组（采样点 + PRN + 时间顺序 + 固定步长）
# ======================
def load_gnss_data(csv_path, time_gap_threshold=1, fixed_time_steps=100, stride=50, features=None):
    if features is None:
        raise ValueError("必须提供 features 列表")
    
    df = pd.read_csv(csv_path)

    print(f"原始数据总量: {len(df)}条记录")
    print("原始类别分布:")
    print(df['Type'].value_counts().sort_index())

    X = []
    y = []

    # Step 1: 按坐标分组
    for (lat, lon), loc_group in df.groupby(['GT_Latitude', 'GT_Longitude']):
    # for site, loc_group in df.groupby('Site'):
        print(f"\nProcessing {(lat, lon)}")
    
        # Step 2: 按 PRN 分组
        for prn, prn_group in loc_group.groupby('PRN'):
            prn_group = prn_group.sort_values('GPS_Time')            
            
            total_length = len(prn_group)
            num_samples = (total_length - fixed_time_steps) // stride + 1

            for i in range(num_samples):
                start_idx = i * stride
                end_idx = start_idx + fixed_time_steps
                sample_df = prn_group.iloc[start_idx:end_idx]

                # For Random Forest, we'll flatten the time steps
                flattened_features = sample_df[features].values.flatten()
                # Use majority voting for the label
                majority_label = sample_df['Type'].mode()[0]
                
                X.append(flattened_features)
                y.append(majority_label)

    # 构造 X 和 y
    X = np.array(X)
    y = np.array(y, dtype=int)  # 强制转换为整型

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\n重组后数据形状: {X.shape}")
    print(f"共生成 {len(X)} 个样本")
    print(f"每个样本特征数: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")

    return X, y, scaler

# ======================
# 训练与评估流程
# ======================
def train_and_evaluate(X, y, features, class_labels):
    # 数据划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # 获取类别数量和名称
    n_classes = len(class_labels)
    target_names = [class_labels[i] for i in sorted(class_labels.keys())]
    label_list = sorted(class_labels.keys())

    # 构建模型
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )

    # 开始训练
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    # 预测并评估
    y_val_pred = model.predict(X_val)

    print("\n=== TABLE 1.EVALUATION METRICS ===")
    print(classification_report(y_val, y_val_pred,
                               target_names=target_names,
                               digits=4))

    print(f"Macro F1: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_val_pred, labels=label_list)
    plot_confusion_matrix(cm, classes=target_names)

    # 特征重要性分析
    analyze_feature_importance(model, X_val, y_val, features, fixed_time_steps)

    return model

def analyze_feature_importance(model, X_val, y_val, feature_names, fixed_time_steps):
    print("\n=== Feature Importance ===")
    
    # 1. Mean Decrease Impurity (内置特征重要性)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names * fixed_time_steps, 
                                      'importance': importances})
    
    # 按原始特征名分组求平均
    grouped_importance = feature_importance.groupby('feature').mean().sort_values('importance', ascending=False)
    print("\nMean Decrease Impurity Importance:")
    print(grouped_importance)
    
    plt.figure(figsize=(8, 6))
    grouped_importance.plot(kind='barh', color='skyblue')
    plt.title("Feature Importance (Mean Decrease Impurity)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
    
    # 2. Permutation Importance
    print("\nCalculating Permutation Importance...")
    result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_importance = pd.DataFrame({
        'feature': feature_names * fixed_time_steps,
        'importance': result.importances_mean
    })
    
    # 按原始特征名分组求平均
    grouped_perm_importance = perm_importance.groupby('feature').mean().sort_values('importance', ascending=False)
    print("\nPermutation Importance:")
    print(grouped_perm_importance)
    
    plt.figure(figsize=(8, 6))
    grouped_perm_importance.plot(kind='barh', color='lightgreen')
    plt.title("Feature Importance (Permutation)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# ======================
# 主执行流程
# ======================
if __name__ == "__main__":
    DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/output/data_b.csv"
    
    FIXED_TIME_STEPS = 30
    STRIDE = 25
   
    FEATURES = ['Elevation','CNR','CNR_rate','Diff_DelPr_PrRate']
    
    CLASS_LABELS = {
        0:'Multi_Path',
        1:'LOS',
        2:'Reflection',
        3:'Diffraction'
    }

    print("Loading data...")
    X, y, scaler = load_gnss_data(DATA_PATH, fixed_time_steps=FIXED_TIME_STEPS, stride=STRIDE, features=FEATURES)
    
    print("\nStart training...")
    model = train_and_evaluate(X, y, features=FEATURES, class_labels=CLASS_LABELS)

    # 保存模型和scaler
    import joblib
    joblib.dump(model, "gnss_signal_classifier_rf.pkl")
    joblib.dump(scaler, "scaler_rf.pkl")
    print("\nModel saved as: gnss_signal_classifier_rf.pkl")
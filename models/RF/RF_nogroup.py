import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import itertools
import warnings
from sklearn.utils.class_weight import compute_class_weight
# 忽略警告
warnings.filterwarnings("ignore")

# ======================
# 统计特征提取函数
# ======================
def extract_statistical_features(df, features):
    stats = {}
    for feature in features:
        stats[f"{feature}_mean"] = df[feature].mean()
        stats[f"{feature}_std"] = df[feature].std()
        stats[f"{feature}_max"] = df[feature].max()
        stats[f"{feature}_min"] = df[feature].min()
        stats[f"{feature}_median"] = df[feature].median()
    return pd.Series(stats)

# ======================
# 数据加载与分组（按站点 + PRN 分组，提取统计特征）
# ======================
def load_gnss_data(csv_path, features=None):
    if features is None:
        raise ValueError("必须提供 features 列表")
    
    df = pd.read_csv(csv_path)
    
    print(f"原始数据总量: {len(df)}条记录")
    print("原始类别分布:")
    print(df['Type'].value_counts().sort_index())
    
    X = []
    y = []

    # Step 1: 按坐标分组（可替换为 Site 分组）
    # for (lat, lon), loc_group in df.groupby(['GT_Latitude', 'GT_Longitude']):
    for site, loc_group in df.groupby('Site'):
        print(f"\nProcessing location {site}")

        # Step 2: 按 PRN 分组
        for prn, prn_group in loc_group.groupby('PRN'):
            prn_group = prn_group.sort_values('GPS_Time')

            # 提取统计特征
            stat_features = extract_statistical_features(prn_group, features)

            # 添加到 X 和 y
            X.append(stat_features)
            y.append(prn_group['Type'].mode()[0])  # 多数投票获取标签

    # 构造 X 和 y
    X = pd.DataFrame(X)
    y = np.array(y, dtype=int)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n重组后数据形状: {X_scaled.shape}")
    print(f"共生成 {len(X_scaled)} 个样本")
    print(f"每个样本特征数: {X_scaled.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")

    return X_scaled, y, scaler

# ======================
# 混淆矩阵绘制函数
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
# 特征重要性分析函数
# ======================
def analyze_feature_importance(model, X_val, y_val, feature_names):
    print("\n=== Feature Importance ===")
    feature_names = [
        f"{feature}_{stat}"
        for feature in feature_names
        for stat in ("mean", "std", "max", "min", "median")
    ]
    
    # Mean Decrease Impurity
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    print("\nMean Decrease Impurity Importance:")
    print(feature_importance)
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title("Feature Importance (Mean Decrease Impurity)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
    
    # Permutation Importance
    print("\nCalculating Permutation Importance...")
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean
    })
    perm_importance = perm_importance.sort_values(by='importance', ascending=False)
    print("\nPermutation Importance:")
    print(perm_importance)
    
    plt.figure(figsize=(8, 6))
    plt.barh(perm_importance['feature'], perm_importance['importance'])
    plt.title("Feature Importance (Permutation)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# ======================
# 训练与评估流程
# ======================
def train_and_evaluate(X, y, features, class_labels):
    # 数据划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 获取类别数量和名称
    n_classes = len(class_labels)
    target_names = [class_labels[i] for i in sorted(class_labels.keys())]
    label_list = sorted(class_labels.keys())

    # 类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

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
    analyze_feature_importance(model, X_val, y_val, features)

    return model

# ======================
# 主执行流程
# ======================
if __name__ == "__main__":
    DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/output/data_b.csv"
    
    FEATURES = ['Elevation', 'CNR', 'CNR_rate', 'Diff_DelPr_PrRate']
    
    CLASS_LABELS = {
        0:'Multi_Path',
        1:'LOS',
        2:'Reflection',
        3:'Diffraction'
    }

    print("Loading data...")
    X, y, scaler = load_gnss_data(DATA_PATH, features=FEATURES)
    
    print("\nStart training...")
    model = train_and_evaluate(X, y, features=FEATURES, class_labels=CLASS_LABELS)

    # 保存模型和scaler
    import joblib
    joblib.dump(model, "gnss_signal_classifier_rf.pkl")
    joblib.dump(scaler, "scaler_rf.pkl")
    print("\nModel saved as: gnss_signal_classifier_rf.pkl")

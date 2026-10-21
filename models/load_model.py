import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools

# ======================
# 全局变量（必须和训练时一致！）
# ======================
FIXED_TIME_STEPS = 30
STRIDE = 25
FEATURES = ['Elevation', 'CNR', 'CNR_rate', 'Diff_DelPr_PrRate']
CLASS_LABELS = {
    0: 'Multi_Path',
    1: 'LOS',
    2: 'Reflection',
    3: 'Diffraction'
}
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

def focal_loss(gamma=2., alpha=[1.0, 1.0, 1.0, 1.0]):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # 获取类别数（来自 y_pred 最后一维）
        num_classes = K.int_shape(y_pred)[-1]
        one_hot_labels = K.one_hot(tf.cast(y_true, tf.int32), num_classes)

        weights = tf.gather(alpha, tf.cast(y_true, tf.int32))

        pt = tf.reduce_sum(one_hot_labels * y_pred, axis=-1)
        pt = K.clip(pt, K.epsilon(), 1. - K.epsilon())
        # 计算损失，并加权
        per_pixel_loss = -K.mean(weights * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)
        return K.mean(per_pixel_loss)
    return focal_loss_fixed
# ======================
# 数据预处理函数（提取特征 + 标签）
# ======================
def preprocess_new_data(csv_path, features=None):
    if features is None:
        raise ValueError("必须提供 features 列表")
    
    df = pd.read_csv(csv_path)
    print(f"原始数据总量: {len(df)}条记录")

    X = []
    y = []

    # Step 1: 按坐标分组
    for (lat, lon), loc_group in df.groupby(['GT_Latitude', 'GT_Longitude']):
        print(f"\nProcessing location ({lat:.5f}, {lon:.5f})")

        # Step 2: 按 PRN 分组
        for prn, prn_group in loc_group.groupby('PRN'):
            prn_group = prn_group.sort_values('GPS_Time')            
            total_length = len(prn_group)
            num_samples = (total_length - FIXED_TIME_STEPS) // STRIDE + 1

            for i in range(num_samples):
                start_idx = i * STRIDE
                end_idx = start_idx + FIXED_TIME_STEPS
                sample_df = prn_group.iloc[start_idx:end_idx]

                X.append(sample_df[features].values)
                y.append(sample_df['Type'].values)

    # 构造输入格式
    X = np.array(X)
    y = np.array(y, dtype=int)

    # 标准化（使用训练集 fit 的 scaler 更准确）
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(original_shape)

    print(f"\n重组后数据形状: {X.shape}")
    print(f"共生成 {len(X)} 个样本组")
    print(f"每个样本时间步长: {FIXED_TIME_STEPS}")
    print(f"类别分布: {np.bincount(y.flatten())}")

    return X, y, df


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
# 主程序入口
# ======================
if __name__ == "__main__":
    DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/Dataset/DataSet_1.csv"

    print("Loading and preprocessing new data...")
    X_new, y_new, raw_df = preprocess_new_data(DATA_PATH, features=FEATURES)

    print("\nLoading trained model...")
    # model = load_model('C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/Bi_LSTM/gnss_signal_classifier_final.keras')
    model = load_model('C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/Bi_LSTM/gnss_signal_classifier_final.keras',
                   custom_objects={'focal_loss_fixed': focal_loss()})

    print("\nMaking predictions...")
    y_pred_probs = model.predict(X_new)
    y_pred = np.argmax(y_pred_probs, axis=-1)  # shape: (num_samples, time_steps)

    # 多数投票获取最终预测标签
    y_true_flat = [np.bincount(seq).argmax() for seq in y_new]
    y_pred_flat = [np.bincount(seq).argmax() for seq in y_pred]

    y_true_flat = np.array(y_true_flat)
    y_pred_flat = np.array(y_pred_flat)

    print("\n=== TABLE 1.EVALUATION METRICS ===")
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    print(classification_report(y_true_flat, y_pred_flat,
                               target_names=target_names,
                               digits=4))

    print(f"Macro F1: {f1_score(y_true_flat, y_pred_flat, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true_flat, y_pred_flat, average='weighted'):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=sorted(CLASS_LABELS.keys()))
    plot_confusion_matrix(cm, classes=target_names)

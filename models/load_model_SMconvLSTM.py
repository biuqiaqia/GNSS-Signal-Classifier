from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from Rho_module import SMConvLSTM  # 必须确保这个模块在路径中
import warnings
warnings.filterwarnings('ignore')


# ======================
# 自定义损失和度量函数（必须重新定义才能加载模型）
# ======================
def masked_loss(y_true, y_pred):
    num_classes = 4
    mask = tf.cast(y_true < num_classes, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        tf.clip_by_value(y_true, 0, num_classes - 1),
        y_pred
    )
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.maximum(tf.reduce_sum(mask), 1e-8)

def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true < 4, tf.float32)
    pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(y_true, pred_classes), tf.float32)
    correct_masked = correct * mask
    return tf.reduce_sum(correct_masked) / tf.maximum(tf.reduce_sum(mask), 1)



# ======================
# 数据预处理函数（与训练时一致）
# ======================
def preprocess_new_data(csv_path, features, fixed_time_steps=50, stride=50):
    def create_sliding_window_samples(X, mask, fixed_time_steps=30, stride=15):
        windows_X = []
        windows_mask = []

        num_epochs = X.shape[0]
        for start in range(0, num_epochs - fixed_time_steps + 1, stride):
            end = start + fixed_time_steps
            window_X = X[start:end]
            window_mask = mask[start:end]

            windows_X.append(window_X)
            windows_mask.append(window_mask)

        return np.array(windows_X), np.array(windows_mask)

    df = pd.read_csv(csv_path)
    if 'Site' not in df.columns:
        df['Site'] = 1
    print(f"原始新数据总量: {len(df)}条记录")
    print("类别分布（如果存在 Uncertainty 列）:")
    if 'Uncertainty' in df.columns:
        print(df['Uncertainty'].value_counts().sort_index())

    all_X = []
    all_mask = []
    all_original_indices = []  # 用于记录每个窗口对应的时间和 PRN

    for site, site_df in df.groupby('Site'):
        print(f"\n处理站点: {site}")
        site_df = site_df.sort_values('GPS_Time')
        time_points = site_df['GPS_Time'].drop_duplicates().sort_values().values 
        num_epochs = len(time_points)

        num_prns = 32
        num_features = len(features)

        X = np.full(shape=(num_epochs, num_prns, num_features), fill_value=np.nan)
        mask = np.zeros(shape=(num_epochs, num_prns), dtype=np.uint8)
        time_to_idx = {time: idx for idx, time in enumerate(time_points)}

        for _, row in site_df.iterrows():
            prn = int(row['PRN']) - 1
            if prn < 0 or prn >= num_prns:
                raise ValueError(f"Invalid PRN: {prn + 1} in Site {site}")


            gps_time = row['GPS_Time']
            time_idx = time_to_idx[gps_time]

            feat_values = row[features].values.astype(np.float32)

            X[time_idx, prn, :] = feat_values
            mask[time_idx, prn] = 1

        # 标准化（仅对有效值）
        scaler = StandardScaler()
        valid_mask = ~np.isnan(X)
        flat_valid = X[valid_mask].reshape(-1, num_features)
        scaled_flat = scaler.fit_transform(flat_valid)
        X[valid_mask] = scaled_flat.flatten()
        X[np.isnan(X)] = 0  # 填充 NaN 为 0

        # 创建滑动窗口
        print(f"站点 {site}: 构造滑动窗口样本（T={fixed_time_steps}, stride={stride}）...")
        X_windows, mask_windows = create_sliding_window_samples(
            X, mask,
            fixed_time_steps=fixed_time_steps,
            stride=stride
        )

        # 扩展维度以匹配模型输入 (T, 32, F) -> (T, 32, 1, F)
        X_windows = X_windows.reshape(-1, fixed_time_steps, 32, 1, num_features)
        mask_windows = mask_windows.reshape(-1, fixed_time_steps, 32, 1)

        all_X.append(X_windows)
        all_mask.append(mask_windows)

        # 记录原始索引信息（可选，用于后续映射回原始观测）
        # 记录原始索引信息（与 X_windows 严格对齐）
        for i in range(X_windows.shape[0]):
            start_idx = i * stride
            end_idx = start_idx + fixed_time_steps
            start_time = time_points[start_idx]
            end_time = time_points[end_idx - 1]
            all_original_indices.append({
                'Site': site,
                'Start_Time': start_time,
                'End_Time': end_time,
                'Window_Index': len(all_original_indices)
            })
    # 合并所有站点
    X_all = np.concatenate(all_X, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)
    print(f"\n最终输入数据形状: X={X_all.shape}, Mask={mask_all.shape}")

    return X_all, mask_all, all_original_indices, df  # 最后返回原始 df



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
# 配置参数
# ======================
MODEL_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/gnss_masked_convlstm_model.h5"
NEW_DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/output/ublox_2_tag.csv"  # 替换为你的新数据路径
OUTPUT_PREDICTION_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/output/aaaaa.csv"   # 保存预测结果的路径

FIXED_TIME_STEPS = 50
STRIDE = 50
FEATURES = ['Elevation', 'Azimuth', 'CNR']
# CLASS_LABELS = {
#     0: 'Multi_Path',
#     1: 'LOS',
#     2: 'Reflection',
#     3: 'Diffraction'
# }
CLASS_LABELS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3'
}

# ======================
# 加载模型（需自定义对象）
# ======================
print("加载训练好的模型...")
custom_objects = {
    'masked_loss': masked_loss,
    'masked_accuracy': masked_accuracy,
    'SMConvLSTM': SMConvLSTM  # 确保 Rho_module 正确导入
}

model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=True)
print("模型加载成功！")


# ======================
# 预测新数据 + 画混淆矩阵
# ======================
print("预处理新数据...")
X_new, mask_new, original_info, raw_df = preprocess_new_data(
    NEW_DATA_PATH,
    features=FEATURES,
    fixed_time_steps=FIXED_TIME_STEPS,
    stride=STRIDE
)

if X_new.shape[0] == 0:
    print("❌ 没有生成任何有效样本，请检查新数据格式和 PRN 范围。")
else:
    print("开始预测...")
    y_pred_probs = model.predict([X_new, mask_new], verbose=1)
    print("y_pred_probs shape before squeeze:", y_pred_probs.shape)
    y_pred_probs = y_pred_probs.squeeze(axis=-2)  # (N, T, 32, 4)
    print("y_pred_probs shape after squeeze:", y_pred_probs.shape)
    y_pred_classes = np.argmax(y_pred_probs, axis=-1)  # (N, T, 32)

    # ======================
    # 创建预测字典: (site, time, prn) -> predicted_label
    # ======================
    pred_dict = {}

    for window_idx in range(y_pred_classes.shape[0]):
        site = original_info[window_idx]['Site']
        start_time = original_info[window_idx]['Start_Time']
        end_time = original_info[window_idx]['End_Time']

        # 从原始数据中提取该站点在该时间段内的所有时间点
        site_df = raw_df[raw_df['Site'] == site]
        time_mask = (site_df['GPS_Time'] >= start_time) & (site_df['GPS_Time'] <= end_time)
        window_times = site_df[time_mask]['GPS_Time'].drop_duplicates().sort_values().values

        if len(window_times) < FIXED_TIME_STEPS:
            continue
        window_times = window_times[:FIXED_TIME_STEPS]  # 取前 50 个

        pred_seq = y_pred_classes[window_idx]  # (50, 32)

        for t in range(FIXED_TIME_STEPS):
            if t >= len(window_times):
                break
            time_point = window_times[t]
            for prn_idx in range(32):
                prn = prn_idx + 1
                pred_label = pred_seq[t, prn_idx].item()
                pred_dict[(site, time_point, prn)] = pred_label

    # ======================
    # 将预测结果写入原始 DataFrame
    # ======================
    raw_df['PRN'] = pd.to_numeric(raw_df['PRN'], errors='coerce')

    # 添加 Predicted_Type 列
    raw_df['Predicted_Type'] = raw_df.apply(
        lambda row: pred_dict.get((row['Site'], row['GPS_Time'], row['PRN']), np.nan),
        axis=1
    )

    # 添加文本标签（可选）
    raw_df['Predicted_Label'] = raw_df['Predicted_Type'].map(CLASS_LABELS).fillna('Unknown')

    # 保存结果
    raw_df.to_csv(OUTPUT_PREDICTION_PATH, index=False)
    print(f"Prediction complete. Results saved to: {OUTPUT_PREDICTION_PATH}")

    # ======================
    # 评估：画混淆矩阵（如果存在真实标签）
    # ======================
    if 'Uncertainty' in raw_df.columns:
        eval_df = raw_df.dropna(subset=['Predicted_Type', 'Uncertainty'])
        if len(eval_df) == 0:
            print("No valid labelled observations are available for evaluation.")
        else:
            y_true = eval_df['Uncertainty'].astype(int)
            y_pred = eval_df['Predicted_Type'].astype(int)

            print("\n=== TABLE 1. EVALUATION METRICS ===")
            target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
            print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

            print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
            print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")

            cm = confusion_matrix(y_true, y_pred, labels=sorted(CLASS_LABELS.keys()))
            plot_confusion_matrix(cm, classes=target_names)
    else:
        print("No 'Uncertainty' column is available for evaluation.")

    # 显示部分结果
    print("\n示例预测结果:")
    print(raw_df[['Site', 'GPS_Time', 'PRN', 'Uncertainty', 'Predicted_Type', 'Predicted_Label']].head(10))

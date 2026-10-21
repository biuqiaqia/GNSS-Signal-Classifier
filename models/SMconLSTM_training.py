import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import itertools
import warnings

from sklearn.inspection import permutation_importance

from sklearn.base import clone
import numpy as np
from tqdm import tqdm


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from Rho_module import SMConvLSTM
def load_gnss_data(csv_path, features, fixed_time_steps=30, stride=15):
    def create_sliding_window_samples(X, y, mask, fixed_time_steps=30, stride=15):
        windows_X = []
        windows_y = []
        windows_mask = []

        num_epochs = X.shape[0]
        for start in range(0, num_epochs - fixed_time_steps + 1, stride):
            end = start + fixed_time_steps
            window_X = X[start:end]  # shape: (T, 32, F)
            window_y = y[start:end]    # shape: (T, 32)
            window_mask = mask[start:end]

            windows_X.append(window_X)
            windows_y.append(window_y)
            windows_mask.append(window_mask)

        return (
            np.array(windows_X),
            np.array(windows_y),
            np.array(windows_mask)
        )

    df = pd.read_csv(csv_path)
    print(f"原始数据总量: {len(df)}条记录")
    print("原始类别分布:")
    print(df['Uncertainty'].value_counts().sort_index())

    all_X = []
    all_y = []
    all_mask = []

    for site, site_df in df.groupby('Site'):
        print(f"\nProcessing Site: {site}")
        
        # 确保时间排序是升序（从早到晚）
        site_df['GPS_Time'] = pd.to_datetime(site_df['GPS_Time'])
        site_df = site_df.sort_values('GPS_Time')
        time_points = site_df['GPS_Time'].unique()
        num_epochs = len(time_points)

        num_prns = 32
        num_features = len(features)
        X = np.full(shape=(num_epochs, num_prns, num_features), fill_value=np.nan)
        y = np.full(shape=(num_epochs, num_prns), fill_value=-1)
        y = np.where(y == -1, 4, y)
        mask = np.zeros(shape=(num_epochs, num_prns), dtype=np.uint8)

        time_to_idx = {time: idx for idx, time in enumerate(time_points)}

        # 填充数据
        for _, row in site_df.iterrows():
            prn = int(row['PRN']) - 1  # 0-based index
            if prn < 0 or prn >= num_prns:
                raise ValueError(f"Invalid PRN: {prn + 1} in Site {site}")

            gps_time = row['GPS_Time']
            time_idx = time_to_idx[pd.to_datetime(gps_time)]

            feat_values = row[features].values.astype(np.float32)
            label_value = int(row['Uncertainty'])

            X[time_idx, prn, :] = feat_values
            y[time_idx, prn] = label_value
            mask[time_idx, prn] = 1

        # 标准化特征（仅对有效数据）
        scaler = StandardScaler()
        valid_mask = ~np.isnan(X)
        flat_values = X[valid_mask].reshape(-1, len(features))
        scaled_flat = scaler.fit_transform(flat_values)
        X[valid_mask] = scaled_flat.flatten()
        X[~valid_mask] = 0  # 将无效位置设为0

        # 构造滑动窗口样本
        print(f"Site {site}: 构造滑动窗口样本（T={fixed_time_steps}, stride={stride}）...")
        X_windows, y_windows, mask_windows = create_sliding_window_samples(
            X, y, mask,
            fixed_time_steps=fixed_time_steps,
            stride=stride
        )
        X_windows = X_windows.reshape(-1, fixed_time_steps, 32, 1, num_features)
        y_windows = y_windows.reshape(-1, fixed_time_steps, 32, 1)
        mask_windows = mask_windows.reshape(-1, fixed_time_steps, 32, 1)
        print(f"Site {site} 生成样本数量: {X_windows.shape[0]}")

        # 收集所有样本
        all_X.append(X_windows)
        all_y.append(y_windows)
        all_mask.append(mask_windows)

    # 合并所有站点的数据
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)


    print("\n最终数据合并完成。")
    print("X_all shape:", X_all.shape)
    print("y_all shape:", y_all.shape)
    print("mask_all shape:", mask_all.shape)
    # print("X_all", X_all)
    # print("y_all", y_all)
    # print("mask_all", mask_all)

    return X_all, y_all, mask_all

def build_model(input_shape, spatial_mask_shape, num_classes=4):
    inputs = Input(shape=input_shape)  # (T, 32, F)
    Smask = Input(shape=spatial_mask_shape)  # (T, 32)

    # conv_lstm = SpatialMaskedConvLSTM(
    #     rank=1,
    #     filters=16,
    #     kernel_size=3,
    #     padding="same",
    #     return_sequences=True,
    #     use_bias=True,
    #     activation="tanh",
    #     recurrent_activation="sigmoid",
    #     data_format="channels_last",
    # )(inputs)  
    
    conv_lstm = SMConvLSTM(
        rank=2,
        filters=16,
        kernel_size=3,
        padding="same",
        return_sequences=True,
        use_bias=True,
        activation="tanh",
        recurrent_activation="sigmoid",
        data_format="channels_last",
    )(inputs, Smask=Smask)  
    outputs = TimeDistributed(Dense(num_classes, activation='softmax'))(conv_lstm)
    
    model = Model(inputs=[inputs,Smask], outputs=outputs)
    # 使用忽略无效标签的损失函数
    # def masked_loss(y_true, y_pred):
    #     mask = tf.math.not_equal(y_true, -1)
    #     loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    #     mask = tf.cast(mask, loss.dtype)
    #     return loss * mask

    def masked_loss(y_true, y_pred):
        # 创建掩码（有效位置=1，无效位置=0）
        mask = tf.cast(y_true < num_classes, tf.float32)
        
        # 计算交叉熵损失（会自动广播mask到与loss相同的形状）
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.clip_by_value(y_true, 0, num_classes-1),  # 确保 y_true ∈ [0,3]
            y_pred
        )
    
        # 应用掩码并求均值
        masked_loss = loss * mask
        return tf.reduce_sum(masked_loss) / tf.maximum(tf.reduce_sum(mask), 1e-8)  # 避免除以0
    
    def masked_accuracy(y_true, y_pred):
        # 确保y_true是整数类型
        y_true = tf.cast(y_true, tf.int32)
        
        mask = tf.cast(y_true < num_classes, tf.float32)
        
        # 计算预测类别
        pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        # 计算正确预测数（只计算有效位置）
        correct = tf.cast(tf.equal(y_true, pred_classes), tf.float32)
        correct_masked = correct * mask
        
        return tf.reduce_sum(correct_masked) / tf.maximum(tf.reduce_sum(mask), 1)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=masked_loss,  # 使用自定义损失函数
        metrics=[masked_accuracy]
    )


    return model

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
# 主执行流程
# ======================
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/output/ublox_3_tag.csv"
    FIXED_TIME_STEPS = 50
    STRIDE = 50
    FEATURES = ['Elevation','Azimuth','CNR']

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
    print("加载数据...")    
    X, y, mask = load_gnss_data(DATA_PATH, 
                               features=FEATURES,
                               fixed_time_steps=FIXED_TIME_STEPS, 
                               stride=STRIDE)
    
    
    
    print("构建模型...")
    input_shape = X.shape[1:]  # (T, 32, F)
    spatial_mask_shape = mask.shape[1:]  # (T, 32)
    num_classes = len(CLASS_LABELS)

    model = build_model(input_shape, spatial_mask_shape, num_classes)
    model.summary()
    weight_mask = np.where(y != -1, 1.0, 0.0)
    print("划分训练集和验证集...")
    (X_train, X_val, 
     y_train, y_val, 
     mask_train, mask_val,
     weight_train, weight_val) = train_test_split(
        X, y, mask, weight_mask, test_size=0.2, shuffle=True
    )

    print("开始训练...")
    history = model.fit(
        [X_train, mask_train], y_train,
        sample_weight=weight_train,  # 添加样本权重
        validation_data=([X_val, mask_val], y_val, weight_val),
        batch_size=32,
        epochs=100
    )
    print("模型训练完成。")

    print("保存模型...")
    model.save("C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/gnss_masked_convlstm_model.h5")
    print("模型已保存为: gnss_masked_convlstm_model.h5")


     # 获取类别数量和名称
    n_classes = len(CLASS_LABELS)
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    label_list = sorted(CLASS_LABELS.keys())




    # 回调函数（Patience，restore）
    callbacks = [
        EarlyStopping(patience=30, monitor='val_accuracy', mode='max', restore_best_weights=True),
        ModelCheckpoint('best_gnss_model.h5', save_best_only=True, monitor='val_accuracy')
    ]

    # 预测并评估
    y_val_pred_probs = model.predict([X_val, mask_val])
    y_val_pred = np.argmax(y_val_pred_probs, axis=-1)  # 在 feature 维度选最大

    # 展平所有时间步
    y_val_flat = y_val.flatten()
    y_val_pred_flat = y_val_pred.flatten()
    valid_indices = valid_indices = y_val_flat < num_classes

    print("\n=== TABLE 1.EVALUATION METRICS ===")
    #print("分类报告:")
    print(classification_report(y_val_flat[valid_indices], y_val_pred_flat[valid_indices],
                               target_names=target_names,
                               digits=4))

   # print("\n关键指标:")
    print(f"Macro F1: {f1_score(y_val_flat[valid_indices], y_val_pred_flat[valid_indices], average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_val_flat[valid_indices], y_val_pred_flat[valid_indices], average='weighted'):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_val_flat[valid_indices], y_val_pred_flat[valid_indices], labels=label_list)
    plot_confusion_matrix(cm, classes=target_names)

    # 绘图：loss & accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
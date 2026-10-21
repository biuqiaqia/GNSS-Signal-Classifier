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

# 忽略警告
warnings.filterwarnings("ignore")

# 自定义 Focal Loss（处理类别不平衡）
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


# confusion_matrix
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


# 1. 数据加载与分组（采样点 + PRN + 时间顺序 + 固定步长）
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
        print(f"\nProcessing {lat}, {lon}")
    
        # Step 2: 按 PRN 分组
        for prn, prn_group in loc_group.groupby('PRN'):
            prn_group = prn_group.sort_values('GPS_Time')            
           
            total_length = len(prn_group)
            num_samples = (total_length - fixed_time_steps) // stride + 1

            for i in range(num_samples):
                start_idx = i * stride
                end_idx = start_idx + fixed_time_steps
                sample_df = prn_group.iloc[start_idx:end_idx]

                X.append(sample_df[features].values)
                y.append(sample_df['Type'].values)

    # 构造 X 和 y
    X = np.array(X)
    y = np.array(y, dtype=int)  

    # 全局标准化（整个训练集 fit 一次）
    scaler = StandardScaler()
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-1])
    X = scaler.fit_transform(X)
    X = X.reshape(original_shape)

    print(f"\n重组后数据形状: {X.shape}")
    print(f"共生成 {len(X)} 个样本组")
    print(f"每个样本时间步长: {fixed_time_steps}")
    print(f"类别分布: {np.bincount(y.flatten())}")

    return X, y



# 2. build_bidirectional_lstm_model
def build_bidirectional_lstm_model(input_shape, n_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.5),
        LSTM(32, return_sequences=True),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=focal_loss(),
        metrics=['accuracy']
    )
    return model



# 3. train_and_evaluate
def train_and_evaluate(X, y, features, class_labels):
   
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # class weight
    '''
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_factors = [1.0, 1.0, 1.0, 1.0]
    class_weight_dict = {
        i: class_weights[i] * weight_factors[i] for i in range(len(class_weights))
    }
    '''

    # class_labels
    n_classes = len(class_labels)
    target_names = [class_labels[i] for i in sorted(class_labels.keys())]
    label_list = sorted(class_labels.keys())


    # model
    model = build_bidirectional_lstm_model((X.shape[1], X.shape[2]), len(np.unique(y)))

    # racall（Patience，restore）
    callbacks = [
        EarlyStopping(patience=30, monitor='val_accuracy', mode='max', restore_best_weights=True),
        ModelCheckpoint('best_gnss_model.h5', save_best_only=True, monitor='val_accuracy')
    ]

    # start training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        #class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # predict and evaluate
    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=-1)  # 在 feature 维度选最大

    # flatten
    y_val_flat = y_val.flatten()
    y_val_pred_flat = y_val_pred.flatten()

    print("\n=== 验证集评估结果 ===")
    print("分类报告:")
    print(classification_report(y_val_flat, y_val_pred_flat,
                               target_names=target_names,
                               digits=4))

    print("\n关键指标:")
    print(f"Macro F1: {f1_score(y_val_flat, y_val_pred_flat, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_val_flat, y_val_pred_flat, average='weighted'):.4f}")

    # confusion matrix
    cm = confusion_matrix(y_val_flat, y_val_pred_flat, labels=label_list)
    plot_confusion_matrix(cm, classes=target_names)

    # plot loss & accuracy
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

    # ========== 特征重要性分析 ==========
    
    analyze_feature_importance(model, X_val, y_val, feature_names=features)
    
    return model


def permutation_importance_3d(model, X, y, scoring_func, n_repeats=10, random_state=42):
    """
    支持 3D 输入 (samples, timesteps, features) 的排列重要性分析
    :param model: 训练好的模型
    :param X: 输入数据 (n_samples, n_timesteps, n_features)
    :param y: 真实标签
    :param scoring_func: 评估函数，如 accuracy、f1 等
    :param n_repeats: 每个特征重复打乱的次数
    :param random_state: 随机种子
    :return: 每个特征的重要性值
    """
    rng = np.random.RandomState(random_state)
    y_pred = np.argmax(model.predict(X), axis=-1)
    baseline_score = scoring_func(y.flatten(), y_pred.flatten())
    n_samples, n_timesteps, n_features = X.shape
    importance_scores = np.zeros((n_features, n_repeats))

    for f in range(n_features):
        print(f"Processing feature '{f}'")
        for repeat in range(n_repeats):
            X_permuted = X.copy()
            # 对每个样本，随机打乱该特征在所有时间步的值（可选：按样本打乱）
            for i in range(n_samples):
                X_permuted[i, :, f] = np.random.permutation(X_permuted[i, :, f])
            
            y_pred_permuted = np.argmax(model.predict(X_permuted), axis=-1)
            score = scoring_func(y.flatten(), y_pred_permuted.flatten())
            importance_scores[f, repeat] = baseline_score - score

    return importance_scores.mean(axis=1)
def analyze_feature_importance(model, X_val, y_val, feature_names):
    print("\n=== 特征重要性分析（Permutation Importance）===")
    from sklearn.metrics import accuracy_score
    # 计算排列重要性
    importances = permutation_importance_3d(
        model, X_val, y_val,
        scoring_func=accuracy_score,
        n_repeats=20,  # 在这
        random_state=42
    )

    # 打印结果
    for feature_name, importance in zip(feature_names, importances):
        print(f"{feature_name}: {importance:.5f}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance via Permutation", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ======================
# 4. 主执行流程
# ======================
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    DATA_PATH = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/Dataset/DataSet_1.csv"
    FIXED_TIME_STEPS = 30
    STRIDE = 25
   
    FEATURES = ['Elevation','CNR', 'CNR_rate', 'Diff_DelPr_PrRate']
    #, 'Pr_Residual''Elevation', 
    '''
    Elevation: 0.00000
    CNR: 0.00677
    Pr_Residual: -0.00827
    CNR_rate: 0.00526
    Diff_DelPr_PrRate: 0.00827
    '''
    CLASS_LABELS = {
        0:'Multi_Path',
        1:'LOS',
        2:'Reflection',
        3:'Diffraction'
         
    }

    print("Loading data...")
    X, y = load_gnss_data(DATA_PATH, fixed_time_steps=FIXED_TIME_STEPS, stride=STRIDE, features=FEATURES)
    
    print("\nStart training...")
    model = train_and_evaluate(X, y, features=FEATURES, class_labels=CLASS_LABELS)

    model.save("C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/System_ongoing/Bi_LSTM/gnss_signal_classifier_final.keras")
    print("\nModel saved as: gnss_signal_classifier_final.keras")
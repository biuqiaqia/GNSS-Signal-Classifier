import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight  # 新增导入
import itertools
import matplotlib
from pathlib import Path

# record the start time
start_time = time.time()

#data loading and preprocessing
ds = pd.read_csv(Path(__file__).resolve().parents[1] / 'TagDatabase' / 'TagDatabase.csv')
features = ds[['CNR', 'Elevation']].values
labels = ds['Type'].values

# data normalization
scaler = StandardScaler()
features = scaler.fit_transform(features)  
# force contiguous memory layout
features = np.ascontiguousarray(features)

# Define the number of samples for each class in the training set
train_samples_per_class = {
    0: 20000,  # NLOS
    1: 20000,  # LOS
    2: 1200,  # NLOS_Refl
    3: 4000,  # NLOS_Diff
    4: 1500   # LOS_Diff
}

# Initialize empty lists to store sampled data
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

# Loop through each class and sample the specified number of training samples
for class_label in np.unique(labels):
    class_mask = (labels == class_label)
    X_class = features[class_mask]
    y_class = labels[class_mask]
    
    n_samples = train_samples_per_class.get(class_label, 0)
    
    if n_samples > len(X_class):
        print(f"warning: type {class_label} only has {len(X_class)} samples, can not use {n_samples}, all samples will be used")
        n_samples = len(X_class)
    
    if n_samples == 0:
        print(f"warning: type {class_label} no applied samples, will be ignored")
        continue
    
   
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class,
        train_size=n_samples,
        random_state=42,
        stratify=y_class 
    )
    
    X_train_list.append(X_train_class)
    X_test_list.append(X_test_class)
    y_train_list.append(y_train_class)
    y_test_list.append(y_test_class)

# Combine the results
X_train = np.concatenate(X_train_list)
X_test = np.concatenate(X_test_list)
y_train = np.concatenate(y_train_list)
y_test = np.concatenate(y_test_list)

print("Training set category distribution:", np.bincount(y_train))
print("Test set category distribution:", np.bincount(y_test))

# MLP with cost-sensitive learning 
class CostSensitiveMLP(MLPClassifier):
    def __init__(self, class_weight=None, **kwargs):
        # add initial parameters
        kwargs.setdefault('early_stopping', True)
        kwargs.setdefault('n_iter_no_change', 20)
        kwargs.setdefault('validation_fraction', 0.2) #
        super().__init__(**kwargs)
        self.class_weight = class_weight
    
    def _calc_loss(self, y_true, y_pred):
        loss = super()._calc_loss(y_true, y_pred)
        if self.class_weight is not None:
            weights = np.array([self.class_weight.get(c, 1) for c in y_true])
            return (loss * weights).mean() / weights.mean()  # 
        return loss

# set class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight = {
    0: class_weights[0],  # NLOS
    1: class_weights[1],  # LOS
    2: class_weights[2] * 2,  # NLOS_Refl
    3: class_weights[3] * 3,  # NLOS_Diff
    4: class_weights[4] * 4   # LOS_Diff
}

# create MLP model 
mlp_model = CostSensitiveMLP(
    class_weight=class_weight,
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=256,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    verbose=True  # output training process
)

history = mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, target_names=['NLOS', 'LOS', 'NLOS_Refl', 'NLOS_Diff', 'LOS_Diff']))

print("\n=== F1 scores of various types ===")
print(f"NLOS_Refl F1: {f1_score(y_test, y_pred, labels=[2], average='micro'):.3f}")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.3f}")


# draw learning curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.validation_scores_, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.loss_curve_, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.show()


# draw confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, SIZE=6):
    plt.figure(figsize=(SIZE, SIZE))
    cm2 = cm
    cmsum = cm.sum(axis=1, keepdims=True)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    if normalize:
        cm = np.divide(cm, cmsum, out=np.zeros_like(cm, dtype=float), where=cmsum != 0)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(title, fontsize=15 + 5, y=1.02)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=50, fontsize=12)
    plt.yticks(tick_marks, classes, rotation=50, fontsize=12)

    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = cm[i, j]
        raw_value = cm2[i, j]

        if cmsum[i] == 0 or cmsum[j] == 0:
            plt.fill_between([j - 0.5, j + 0.5], i - 0.5, i + 0.5, color='red', alpha=0.5)
            text = " "
        elif np.isnan(value):
            text = ""
        else:
            text = f"{value:.3f}" if normalize else f"{int(raw_value)}"

        plt.text(j, i, text,
                 horizontalalignment="center", fontsize=12,
                 color="white" if value > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
plot_confusion_matrix(cm, classes=['NLOS', 'LOS', 'NLOS_Refl', 'NLOS_Diff', 'LOS_Diff'])
plt.show()

#  plot decision boundary
def plot_decision_boundary_with_data_counts():
    h = 0.02
    # 
    x_min = min(features[:, 0].min(), X_train[:, 0].min(), X_test[:, 0].min()) - 1.0
    x_max = max(features[:, 0].max(), X_train[:, 0].max(), X_test[:, 0].max()) + 1.0
    y_min = min(features[:, 1].min(), X_train[:, 1].min(), X_test[:, 1].min()) - 1.0
    y_max = max(features[:, 1].max(), X_train[:, 1].max(), X_test[:, 1].max()) + 1.0

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = scaler.transform(grid)
    Z = mlp_model.predict(grid)
    Z = Z.reshape(xx.shape)

    # create figure and axes
    fig = plt.figure(figsize=(22, 10), facecolor='#FFF8E7') 
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 0.1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])  # left
    ax2 = fig.add_subplot(gs[0, 1])  # right
    ax_legend = fig.add_subplot(gs[1, :])  # bottom
    
    # 
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_facecolor('#FFF8E7')
    
    # left map
    contour = ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter_train = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                              edgecolors='w', cmap=plt.cm.RdYlBu, alpha=0.7, s=30)
    ax1.set_xlabel('CNR', fontsize=12)
    ax1.set_ylabel('Elevation', fontsize=12)
    ax1.set_title('Training Data Distribution', pad=15) #
    
    # right map
    scatter_test = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                             edgecolors='w', cmap=plt.cm.RdYlBu, alpha=0.7, s=30)
    ax2.set_xlabel('CNR', fontsize=12)
    ax2.set_ylabel('Elevation', fontsize=12)
    ax2.set_title('Test Data Distribution', pad=15)
    
    # bottom legend
    legend_elements = scatter_train.legend_elements()[0]
    legend_labels = ['NLOS', 'LOS', 'NLOS_Refl', 'NLOS_Diff', 'LOS_Diff']
    legend = ax_legend.legend(legend_elements, legend_labels,
                            loc='center', 
                            ncol=5,
                            frameon=False,
                            title="Click to toggle classes:",
                            title_fontsize=12)
    ax_legend.axis('off')  #
    
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.show()

plot_decision_boundary_with_data_counts()

print(f"\totally {time.time()-start_time:.1f} second")


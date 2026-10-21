clear; clc; 
close all;

% 数据路径
csv_path = 'output\2024Da06_tag.csv';

% 参数设置
fixed_time_steps = 60;
stride = 50;

% Step 1: 加载并预处理数据
disp('Loading and preprocessing data...');
[X, Y] = load_gnss_data(csv_path, fixed_time_steps, stride);

% Step 2: 构建双向 LSTM 模型结构
layers = [
    sequenceInputLayer([6 1 1], 'Name', 'input')  % 输入维度 [特征数 x 1 x 1]
    bilstmLayer(32, 'OutputMode', 'sequence')     % 删除 'MaskValue', 0
    dropoutLayer(0.5)
    bilstmLayer(32, 'OutputMode', 'last')
    dropoutLayer(0.5)
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer
];
% Step 3: 自定义损失函数（Focal Loss）
gamma = 2;
alpha = [1.0, 1.0, 10.0, 5.0, 5.0];
lossFcn = @(Y,T) focal_loss(Y, T, gamma, alpha);

% Step 4: 设置训练选项
% 划分训练集/验证集
idx = randperm(size(X, 3));
trainIdx = idx(1:round(0.8 * length(idx)));
valIdx = idx(round(0.8 * length(idx)) + 1:end);

XTrain = X(:, :, trainIdx);
YTrain = Y(trainIdx);
XVal = X(:, :, valIdx);
YVal = Y(valIdx);

% 构建模型
% 构建模型
layers = [
    sequenceInputLayer([6 1 1], 'Name', 'input')
    bilstmLayer(32, 'OutputMode', 'sequence')  % 删除 MaskValue
    dropoutLayer(0.5)
    bilstmLayer(32, 'OutputMode', 'last')
    dropoutLayer(0.5)
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer('Name', 'output', ...
        'LossFunction', 'crossentropy', ...
        'ClassWeights', [1.0, 1.0, 10.0, 5.0, 5.0])  % 模拟 Focal Loss 权重
];

% 训练参数
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {XVal, YVal}, ...
    'ExecutionEnvironment', 'auto');

% 开始训练
net = trainNetwork(XTrain, YTrain, layers, options);

% Step 7: 验证集预测
YPred = classify(net, XVal);

% Step 8: 打印分类报告
fprintf('\n=== 验证集评估结果 ===\n');
fprintf('分类报告:\n');
classNames = {'NLOS', 'LOS', 'NLOS_Refl', 'NLOS_Diff', 'LOS_Diff'};
classificationReport(YVal, YPred, 'classnames', classNames);

% Step 9: 绘制混淆矩阵
figure;
cm = confusionchart(YVal, YPred, 'ClassNames', classNames);
title('Confusion Matrix', 'FontSize', 14);
cm.Title = 'Validation Set';
cm.RowSummary = 'row-normalized';  % 归一化显示
cm.ColumnSummary = 'column-normalized';

% Step 10: 可选 - 保存模型
save('trained_gnss_model.mat', 'net');
disp('Model saved to trained_gnss_model.mat');
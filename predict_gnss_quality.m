function [pred_Type, pred_Uncertainty] = predict_gnss_quality(gnss_epoch_data, model_path, scaler_params)
% predict_gnss_quality - 调用 Python 模型预测 GNSS 卫星质量
%
% 输入:
%   gnss_epoch_data: GNSS_data{epoch}
%   model_path: 字符串，模型路径
%   scaler_params: 包含 mean 和 std 的结构体
%
% 输出:
%   pred_Type: N×1 向量
%   pred_Uncertainty: N×1 向量

%% 提取数据
prn   = gnss_epoch_data{3};      % PRN
el    = gnss_epoch_data{4};      % 仰角
az    = gnss_epoch_data{5};      % 方位角
cnr   = gnss_epoch_data{9}(:,1); % CNR

% 构造特征矩阵
features = [el, az, cnr];  % N x 3
N = size(features, 1);

% 归一化
feat_mean = scaler_params.mean;  % [1x3]
feat_std  = scaler_params.std;
features_norm = (features - feat_mean) ./ feat_std;
features_norm(isnan(features_norm)) = 0;

% 设置 Python 路径
py.sys.path.insert(int64(0), fileparts(which('predict_gnss_quality.m')));

%% 调用 Python
try
    pred_list = py.predict_gnss.predict_single_epoch(...
        features_norm, ones(N,1), model_path);
catch ME
    error('Python 推理失败: %s', ME.message);
end

%% 后处理
pred_Type = double(pred_list);  % list -> double array
pred_Type = pred_Type(:);      % 列向量

% 当前 Uncertainty 与 Type 相同
pred_Uncertainty = pred_Type;

end
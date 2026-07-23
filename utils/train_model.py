import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. 加载数据
df = pd.read_csv('../data/Super_Cleaned_Concrete_Data.csv')

# --- 改进 A: 特征工程 (Feature Engineering) ---
# 增加物理意义更强的特征，帮助模型理解化学反应基础
df['binder_water_ratio'] = df['TOTAL_BINDER'] / (df['WATER'])
df['PC_ratio'] = df['PC'] / (df['TOTAL_BINDER'] )
df['FA_ratio'] = df['FA'] / (df['TOTAL_BINDER'] )
df['SC_ratio'] = df['SC'] / (df['TOTAL_BINDER'] )

# 自动获取基础特征列表（排除目标列和中间列）
base_features = [col for col in df.columns if col not in ['7day', '28day', '56day']]
print(f"基础特征数量: {len(base_features)} | 包括: {base_features}")

# 2. 设置参数空间
param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1, 10, 100],  # 增加正则化控制
    'min_child_weight': [1, 3, 5]
}


def get_best_model_random(X, y, name):
    print(f"\n>>> 正在寻优 {name} 模型 (样本数: {len(X)})...")
    random_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, tree_method='hist'),
        param_distributions=param_dist,
        n_iter=30,  # 可根据时间调大
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    random_search.fit(X, y)
    print(f"[{name}] 最佳参数: {random_search.best_params_}")
    return random_search.best_estimator_


# --- 改进 B: 训练逻辑 (按目标过滤以保留更多样本) ---
models = {}
train_test_data = {}

# 分别为三个目标准备数据并训练
for target in ['7day', '28day', '56day']:
    # 确定当前阶段的输入特征
    if target == '7day':
        current_features = base_features
    elif target == '28day':
        current_features = base_features + ['7day']
    else:  # 56day
        current_features = base_features + ['28day']

    # 仅针对当前有目标值的行进行训练（不强制要求三者都有值，增加 56 天的训练样本）
    target_df = df.dropna(subset=[target]).copy()
    if target != '7day':
        # 确保链式特征（前一阶段强度）也没有空值
        prev_target = '7day' if target == '28day' else '28day'
        target_df = target_df.dropna(subset=[prev_target])

    X_train, X_test, y_train, y_test = train_test_split(
        target_df[current_features], target_df[target],
        test_size=0.2, random_state=42
    )

    # 训练模型
    best_model = get_best_model_random(X_train, y_train, target)
    models[target] = best_model

    # 保存该目标的测试集，用于最后的模拟真实推理
    train_test_data[target] = (X_train, X_test, y_train, y_test)

# --- 改进 C: 全面精度评估 (Train vs Test) ---
print("\n" + "=" * 50)
print("各阶段模型拟合度评估 (独立评估):")
for target in ['7day', '28day', '56day']:
    X_train, X_test, y_train, y_test = train_test_data[target]
    train_r2 = r2_score(y_train, models[target].predict(X_train))
    test_r2 = r2_score(y_test, models[target].predict(X_test))
    print(f"[{target:5s}] Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")

# --- 5. 模拟真实情况测试 (考虑误差累积) ---
print("\n" + "=" * 50)
print("开始模拟真实预测场景 (链式预测 + 误差累积评估)...")

# 为了模拟真实预测，我们需要一个包含所有阶段真实值的子集进行链式比对
# 这里我们取 test_df（三个值全有的行）
eval_df = df.dropna(subset=['7day', '28day', '56day']).copy()
_, test_real_df = train_test_split(eval_df, test_size=0.2, random_state=42)

# 1. 预测 7天
test_real_df['pred_7day'] = models['7day'].predict(test_real_df[base_features])

# 2. 预测 28天 (使用预测的 7天结果)
X_28_sim = test_real_df[base_features].copy()
X_28_sim['7day'] = test_real_df['pred_7day']
test_real_df['pred_28day'] = models['28day'].predict(X_28_sim)

# 3. 预测 56天 (使用预测的 28天结果)
X_56_sim = test_real_df[base_features].copy()
X_56_sim['28day'] = test_real_df['pred_28day']
test_real_df['pred_56day'] = models['56day'].predict(X_56_sim)

# 6. 计算模拟推理下的 R2
final_metrics = []
for day in ['7day', '28day', '56day']:
    r2 = r2_score(test_real_df[day], test_real_df[f'pred_{day}'])
    mae = mean_absolute_error(test_real_df[day], test_real_df[f'pred_{day}'])
    final_metrics.append({'龄期': day, '链式预测测试集 R2': round(r2, 4), 'MAE': round(mae, 4)})

print("\n最终评估结果 (链式推理场景):")
print(pd.DataFrame(final_metrics))

# 7. 保存
meta = {
    'models': models,
    'feature_names': base_features,
    'engineered_features': ['binder_water_ratio', 'PC_ratio', 'FA_ratio', 'SC_ratio']
}
joblib.dump(meta, 'concrete_optimized_chained.pkl')
print("\n✅ 优化寻优完成！")
import pandas as pd
import numpy as np
import catboost as cb
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. 加载数据
df = pd.read_csv('Super_Cleaned_Concrete_Data.csv')

# --- 修改后的特征工程 (已移除 SF 相关项) ---

# 1. 重新定义 TOTAL_BINDER (仅包含 PC, FA, SC)
df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC']

# 2. 重新定义 w/b 比 (无需修改，只需确保 TOTAL_BINDER 已更新)
df['w/b'] = df['WATER'] / df['TOTAL_BINDER']

# 3. 骨料相关特征保持不变
agg_sum = df['FAGG'] + df['CAGG']
df['b/a'] = df['TOTAL_BINDER'] / agg_sum
df['CAGG%'] = df['CAGG'] / agg_sum
df['FAGG%'] = df['FAGG'] / agg_sum

# 4. SCM% 计算移除 SF
df['SCM%'] = (df['FA'] + df['SC']) / df['TOTAL_BINDER']

# 5. 各组分百分比占比 (移除 SF%)
df['PC%'] = df['PC'] / df['TOTAL_BINDER']
df['FA%'] = df['FA'] / df['TOTAL_BINDER']
df['SC%'] = df['SC'] / df['TOTAL_BINDER']

# 检查一下是否存在多余的列引用
# 注意：确保 base_independent_vars 中也移除了 'SF'
base_independent_vars = ['PC', 'FA', 'SC', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']


# 获取基础特征列表
base_features = [col for col in df.columns if col not in ['7day', '28day', '56day']]

# 2. 设置 CatBoost 参数空间
# CatBoost 的参数名与 XGB 有所不同
param_dist = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 10, 100],  # 相当于 XGB 的 lambda
    'random_strength': [1, 2, 5],  # 样本随机扰动，防止过拟合
    'bagging_temperature': [0, 0.5, 1],  # 控制采样
    'border_count': [128, 254]  # 数值特征的分桶数
}


def get_best_model_cat(X, y, name):
    print(f"\n>>> 正在通过随机搜索寻优 {name} 模型 (CatBoost 引擎)...")

    # CatBoostRegressor 初始化
    # task_type='CPU' 如果有显卡可以改为 'GPU'
    cat_model = cb.CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        logging_level='Silent',
        allow_writing_files=False
    )

    random_search = RandomizedSearchCV(
        cat_model,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    random_search.fit(X, y)
    print(f"[{name}] 最佳参数: {random_search.best_params_}")
    return random_search.best_estimator_


# 3. 链式训练准备
models = {}
train_test_data = {}

# 循环训练三个龄期模型
for target in ['7day', '28day', '56day']:
    if target == '7day':
        current_features = base_features
    elif target == '28day':
        current_features = base_features + ['7day']
    else:  # 56day
        current_features = base_features + ['28day']

    # 按需过滤数据，最大化样本量
    target_df = df.dropna(subset=[target]).copy()
    if target != '7day':
        prev_target = '7day' if target == '28day' else '28day'
        target_df = target_df.dropna(subset=[prev_target])

    X_train, X_test, y_train, y_test = train_test_split(
        target_df[current_features], target_df[target],
        test_size=0.2, random_state=42
    )

    # 寻优训练
    best_model = get_best_model_cat(X_train, y_train, target)
    models[target] = best_model
    train_test_data[target] = (X_train, X_test, y_train, y_test)

# 4. 评估独立性能 (Train vs Test)
print("\n" + "=" * 50)
print("CatBoost 各阶段模型独立拟合度:")
for target in ['7day', '28day', '56day']:
    X_train, X_test, y_train, y_test = train_test_data[target]
    train_r2 = r2_score(y_train, models[target].predict(X_train))
    test_r2 = r2_score(y_test, models[target].predict(X_test))
    print(f"[{target:5s}] Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")

# 5. 模拟真实推理场景 (处理误差累积)
print("\n" + "=" * 50)
print("开始模拟 CatBoost 链式推理 (误差累积测试)...")

eval_df = df.dropna(subset=['7day', '28day', '56day']).copy()
_, test_real_df = train_test_split(eval_df, test_size=0.2, random_state=42)

# 7d -> 28d -> 56d 链式预测
test_real_df['pred_7day'] = models['7day'].predict(test_real_df[base_features])

X_28_sim = test_real_df[base_features].copy()
X_28_sim['7day'] = test_real_df['pred_7day']
test_real_df['pred_28day'] = models['28day'].predict(X_28_sim)

X_56_sim = test_real_df[base_features].copy()
X_56_sim['28day'] = test_real_df['pred_28day']
test_real_df['pred_56day'] = models['56day'].predict(X_56_sim)

# 6. 计算评估指标
final_metrics = []
for day in ['7day', '28day', '56day']:
    r2 = r2_score(test_real_df[day], test_real_df[f'pred_{day}'])
    mae = mean_absolute_error(test_real_df[day], test_real_df[f'pred_{day}'])
    final_metrics.append({'龄期': day, 'CatBoost 链式 R2': round(r2, 4), 'MAE': round(mae, 4)})

print("\n最终评估结果 (CatBoost 推理场景):")
print(pd.DataFrame(final_metrics))

# --- 请用这段代码替换原脚本中的 # 7. 保存模型 部分 ---

# 1. 定义需要反归一化的基础特征列表 (必须和 MOO 算法中的变量一致)
# base_independent_vars = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']

# 2. 从当前使用的 dataframe 中提取这些特征的最小值和最大值
mins_dict = df[base_independent_vars].min().to_dict()
maxs_dict = df[base_independent_vars].max().to_dict()

# 3. 封装到 meta 字典中
meta = {
    'models': models,
    'feature_names': base_features,  # 所有的特征名（含衍生特征）
    'mins': mins_dict,               # 基础特征的最小值
    'maxs': maxs_dict,               # 基础特征的最大值
    'type': 'catboost_chained'
}

# 4. 保存文件
import joblib
joblib.dump(meta, 'concrete_catboost_optimized.pkl')

print("\n" + "="*30)
print("✅ 模型及元数据已完整保存！")
print(f"包含的元数据键值: {list(meta.keys())}")
print(f"包含的归一化基准: {list(meta['mins'].keys())}")
print("="*30)
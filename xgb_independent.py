import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. 加载数据
df = pd.read_csv('Sheet3_Final_Training_Data.csv')

# 基础特征（完全不包含任何强度信息）
base_features = [col for col in df.columns if col not in ['7day', '28day', '56day', 'GWP']]
print(f"输入特征: {base_features}")

# 2. 划分数据集 (确保测试集与之前的实验一致，以便对比)
train_df, test_df = train_test_split(df.dropna(subset=['7day', '28day', '56day']),
                                     test_size=0.2, random_state=42)

# 3. 广域寻优参数配置
param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}


def train_independent_model(target_name):
    print(f"\n>>> 正在独立训练 {target_name} 预测模型...")

    # 这里的 X 始终只有 base_features
    X_train = train_df[base_features]
    y_train = train_df[target_name]

    rs = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, tree_method='hist'),
        param_distributions=param_dist,
        n_iter=40,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rs.fit(X_train, y_train)
    print(f"[{target_name}] 最佳参数: {rs.best_params_}")
    return rs.best_estimator_


# 4. 分别训练三个模型
independent_models = {}
for day in ['7day', '28day', '56day']:
    independent_models[day] = train_independent_model(day)

# 5. 测试与评估 (直接预测)
print("\n" + "=" * 30)
print("开始独立模型性能评估...")

results = []
for day in ['7day', '28day', '56day']:
    # 预测时只用基础特征
    y_pred = independent_models[day].predict(test_df[base_features])

    r2 = r2_score(test_df[day], y_pred)
    mae = mean_absolute_error(test_df[day], y_pred)

    results.append({
        'AGE': day,
        'R2': round(r2, 4),
        'MAE': round(mae, 4)
    })

# 6. 输出结果
results_df = pd.DataFrame(results)
print("\n独立回归测试集评估结果:")
print(results_df)

# 7. 保存
joblib.dump({
    'models': independent_models,
    'features': base_features
}, 'concrete_independent_xgboost.pkl')

print("\n✅ 独立模型训练完成。你可以对比发现 56d 的 R2 通常会比链式法则低，因为少了 28d 这个强力特征。")
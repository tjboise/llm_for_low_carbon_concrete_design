import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import catboost as cb

# 1. 数据准备
df = pd.read_csv('Super_Cleaned_Concrete_Data.csv')
df = df.drop(columns=['SF'], errors='ignore')

# 特征工程
df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC']
df['w/b'] = df['WATER'] / df['TOTAL_BINDER']
agg_sum = df['FAGG'] + df['CAGG']
df['b/a'] = df['TOTAL_BINDER'] / agg_sum
df['SCM%'] = (df['FA'] + df['SC']) / df['TOTAL_BINDER']
df['CAGG%'] = df['CAGG'] / agg_sum
df['FAGG%'] = df['FAGG'] / agg_sum
df['PC%'] = df['PC'] / df['TOTAL_BINDER']
df['FA%'] = df['FA'] / df['TOTAL_BINDER']
df['SC%'] = df['SC'] / df['TOTAL_BINDER']

base_features = [col for col in df.columns if col not in ['7day', '28day', '56day']]
targets = ['7day', '28day', '56day']

# 2. 定义模型（MLP 必须加标准化以保证收敛）
models_config = {
    'RF': RandomForestRegressor(n_estimators=100),
    'CatBoost': cb.CatBoostRegressor(logging_level='Silent'),
    'XGBoost': xgb.XGBRegressor(),
    'MLP': Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000))])
}

# 3. 循环 3 次实验
results = []
for i in range(3):
    print(f"执行第 {i+1}/3 次随机实验...")
    for name, model in models_config.items():
        for target in targets:
            temp_df = df.dropna(subset=[target])
            X_train, X_test, y_train, y_test = train_test_split(
                temp_df[base_features], temp_df[target], test_size=0.2, random_state=42 + i
            )
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            results.append({'Model': name, 'Age': target, 'R2': r2})

results_df = pd.DataFrame(results)

# 4. 可视化 (带误差棒)
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Age', y='R2', hue='Model', capsize=.1, errorbar='sd')

plt.title('Performance Comparison (Mean R2 Score with Standard Deviation)', fontsize=14)
plt.ylabel('R2 Score', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. 保存统计结果
summary = results_df.groupby(['Model', 'Age'])['R2'].agg(['mean', 'std']).reset_index()
print("\n统计结果汇总：")
print(summary)
summary.to_csv('model_comparison_stats.csv', index=False)
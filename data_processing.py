import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score

# 1. 配置参数
INPUT_FILE = 'Sheet3_Final_Training_Data.csv'
OUTPUT_FILE = 'Super_Cleaned_Concrete_Data.csv'
FEATURES = ['PC', 'FA', 'SC', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC', 'TOTAL_BINDER', 'w/b', 'b/a',
            'SCM%', 'CAGG%', 'FAGG%']
TARGETS = ['7day', '28day', '56day']


def generate_super_clean_data():
    # 读取原始数据
    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据量: {len(df)}")

    # --- 步骤 A: 物理逻辑初步筛选 ---
    # 确保强度随时间增长（7d <= 28d <= 56d），删除逻辑错误的行
    mask_logic = ~((df['7day'] > df['28day']) | (df['28day'] > df['56day']) | (df['7day'] > df['56day']))
    df = df[mask_logic.fillna(True)]

    # --- 步骤 B: 多变量异常检测 (Isolation Forest) ---
    iso = IsolationForest(contamination=0.03, random_state=42)
    df['is_outlier'] = iso.fit_predict(df[FEATURES])
    df = df[df['is_outlier'] == 1].drop(columns=['is_outlier'])

    # --- 步骤 C: 重复实验均值化 ---
    # 将相同配方的数据合并取平均，消除单次实验的随机误差
    df = df.groupby(FEATURES, as_index=False)[TARGETS].mean()
    print(f"去重及初步过滤后数据量: {len(df)}")

    # --- 步骤 D: 递归残差清洗 (核心步骤) ---
    # 定义一个内部函数，通过模型训练剔除“不听话”的数据点
    def iterative_residual_cleaning(data, target_col, rounds=2, sigma=2.0):
        temp_df = data.dropna(subset=[target_col]).copy()
        for r in range(rounds):
            X = temp_df[FEATURES]
            y = temp_df[target_col]

            # 使用 XGB 进行基准拟合
            model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
            model.fit(X, y)

            # 计算残差
            preds = model.predict(X)
            residuals = np.abs(y - preds)
            threshold = sigma * residuals.std()

            # 仅保留残差在阈值内的点
            keep_mask = residuals <= threshold
            temp_df = temp_df[keep_mask]
            print(f"目标 {target_col} | 第 {r + 1} 轮清洗: 剩余 {len(temp_df)} 行")
        return temp_df.index.tolist()

    # 执行 7天和 28天强度的双重深度清洗
    valid_7 = iterative_residual_cleaning(df, '7day')
    valid_28 = iterative_residual_cleaning(df, '28day')

    # 取两个关键指标都“表现良好”的交集
    final_indices = set(valid_7).intersection(set(valid_28))
    df_super_clean = df.loc[list(final_indices)]

    # 导出文件
    df_super_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 成功生成! 最终高质量数据量: {len(df_super_clean)}")
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_super_clean_data()
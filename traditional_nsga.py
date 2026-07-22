import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# ==========================================================
# 1. 传统预测大脑：仅包含物理计算与 CatBoost 链式预测
# ==========================================================
class TraditionalBrain:
    def __init__(self, model_pkl, csv_data):
        # A. 加载模型及元数据
        meta = joblib.load(model_pkl)
        self.models = meta['models']
        self.all_feature_names = meta['feature_names']
        self.mins = meta.get('mins', {})
        self.maxs = meta.get('maxs', {})
        self.independent_vars = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']

        # B. 从你的最新 CSV 获取物理边界
        df_train = pd.read_csv(csv_data)
        self.phys_mins = df_train[self.independent_vars].min().values
        self.phys_maxs = df_train[self.independent_vars].max().values

        # 碳排放因子 (对应你代码中的 GWP_FACTORS)
        self.gwp_factors = {'PC': 1.048, 'FA': 0.328, 'SC': 0.264, 'SF': 0.850, 'CAGG': 0.0037, 'FAGG': 0.0026}

    def predict_workflow(self, x_norm):
        """向量化预测逻辑"""
        # 1. 逆归一化
        phys_x = x_norm * (self.phys_maxs - self.phys_mins) + self.phys_mins
        df = pd.DataFrame(phys_x, columns=self.independent_vars)

        # 2. 衍生特征工程 (同步你之前的逻辑)
        df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC'] + df['SF']
        df['w/b'] = df['WATER'] / (df['TOTAL_BINDER'].replace(0, np.nan))
        agg_sum = df['FAGG'] + df['CAGG']
        df['b/a'] = df['TOTAL_BINDER'] / (agg_sum.replace(0, np.nan))
        df['SCM%'] = (df['FA'] + df['SC'] + df['SF']) / (df['TOTAL_BINDER'].replace(0, np.nan))
        df['CAGG%'] = df['CAGG'] / (agg_sum.replace(0, np.nan))
        df['FAGG%'] = df['FAGG'] / (agg_sum.replace(0, np.nan))
        df['PC%'] = df['PC'] / (df['TOTAL_BINDER'].replace(0, np.nan))
        df['FA%'] = df['FA'] / (df['TOTAL_BINDER'].replace(0, np.nan))
        df['SC%'] = df['SC'] / (df['TOTAL_BINDER'].replace(0, np.nan))

        # 3. 链式预测
        X_base = df[self.all_feature_names]
        p7 = self.models['7day'].predict(X_base)

        X_28 = X_base.copy();
        X_28['7day'] = p7
        p28 = np.maximum(self.models['28day'].predict(X_28), p7)

        X_56 = X_base.copy();
        X_56['28day'] = p28
        p56 = np.maximum(self.models['56day'].predict(X_56), p28)

        # 4. GWP 计算
        gwp = np.zeros(len(df))
        for mat, factor in self.gwp_factors.items():
            if mat in df.columns:
                gwp += df[mat].values * factor

        return {'p28': p28, 'p56': p56, 'gwp': gwp, 'df': df}


# ==========================================================
# 2. 优化问题定义
# ==========================================================
class TraditionalProblem(Problem):
    def __init__(self, brain, target_28d):
        self.brain = brain
        self.target_28d = target_28d
        super().__init__(n_var=11, n_obj=2, n_constr=1, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.brain.predict_workflow(x)
        out["F"] = np.column_stack([res['gwp'], -res['p56']])
        out["G"] = np.array(self.target_28d - res['p28'])


# ==========================================================
# 3. TOPSIS 决策与主程序
# ==========================================================
def run_topsis(F, weights=np.array([0.5, 0.5])):
    """
    F: 目标值矩阵 [n, 2], 第一列是 GWP, 第二列是 -p56
    weights: 权重向量，和为 1
    """
    # 1. 归一化 (向量归一化)
    norm_F = F / (np.sqrt((F ** 2).sum(axis=0)) + 1e-8)

    # 2. 加权
    weighted_F = norm_F * weights

    # 3. 确定加权理想解和负理想解
    ideal_best = weighted_F.min(axis=0)  # 越小越好 (GWP小, -p56小即p56大)
    ideal_worst = weighted_F.max(axis=0)

    # 4. 计算欧式距离
    dist_best = np.sqrt(((weighted_F - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_F - ideal_worst) ** 2).sum(axis=1))

    # 5. 计算得分 (离最坏解越远、离最好解越近，分越高)
    score = dist_worst / (dist_best + dist_worst + 1e-8)
    return np.argmax(score)


if __name__ == "__main__":
    BRAIN_FILE = "concrete_catboost_optimized.pkl"
    DATA_FILE = "Super_Cleaned_Concrete_Data.csv"

    print("--- 🚀 初始化传统搜索模型 (无 LLM 审计) ---")
    brain = TraditionalBrain(BRAIN_FILE, DATA_FILE)

    # 1. 定义任务：28d >= 40MPa
    problem = TraditionalProblem(brain, target_28d=40)

    # 2. 运行 NSGA-II 优化
    res = minimize(problem, NSGA2(pop_size=200), ('n_gen', 100), seed=42)

    # 3. 处理结果
    if res.X is not None:
        # 筛选满足强度要求的可行解
        feasible = (res.G <= 0).flatten()

        if not any(feasible):
            print("❌ 未找到满足 28d 强度要求 (40MPa) 的可行解。")
        else:
            F_f, X_f = res.F[feasible], res.X[feasible]

            # --- [重要] 保存 Baseline 数据用于后续对比图 ---
            # 存储格式为：[GWP, 56d强度]
            baseline_to_save = np.column_stack([F_f[:, 0], -F_f[:, 1]])
            np.save("baseline_pareto_data.npy", baseline_to_save)
            print("✅ Baseline Pareto Front 数据已保存至 baseline_pareto_data.npy")

            # 4. TOPSIS 决策 (GWP 与 56d 权重各占 0.5)
            best_idx = run_topsis(F_f, [0.5, 0.5])

            # 5. 获取并还原最优设计的物理值
            final = brain.predict_workflow(X_f[best_idx].reshape(1, -1))
            df_best = final['df'].iloc[0]

            # 提取 7d 强度 (通过模型预测)
            X_base_vals = df_best[brain.all_feature_names].values.reshape(1, -1)
            p7_best = brain.models['7day'].predict(X_base_vals)[0]

            # ========================= 格式化输出 =========================
            print("\n" + "========================= The final design =========================")
            print(f"{'component':<16} | {'dosage (lb/yd³)':<15}")
            print("-" * 45)

            # 打印 11 个原料组分
            for var in brain.independent_vars:
                print(f"{var:<16} | {df_best[var]:>12.2f}")

            print("-" * 45)

            # 打印性能指标
            print(f"{'GWP':<16} | {final['gwp'][0]:>12.2f} lb/m³")
            print(f"{'w/b':<16} | {df_best['w/b']:>12.3f}")
            print(f"{'SCM%':<16} | {df_best['SCM%'] * 100:>11.2f}%")
            print(f"{'7d strength':<16} | {p7_best:>10.2f} MPa")
            print(f"{'28d strength':<16} | {final['p28'][0]:>10.2f} MPa")
            print(f"{'56d strength':<16} | {final['p56'][0]:>10.2f} MPa")
            print("=" * 52)
            # ==============================================================
            baseline_front = np.column_stack([F_f[:, 0], -F_f[:, 1]])
            np.save("baseline_pareto_data.npy", baseline_front)

            # 2. 保存最佳平衡点 (TOPSIS Selected)
            baseline_best_point = np.array([final['gwp'][0], final['p56'][0]])
            np.save("baseline_best_point.npy", baseline_best_point)

            print("✅ Baseline 前沿与最佳点数据已保存。")

            # 6. 绘图 (Y 轴对齐为 56d 强度)
            plt.figure(figsize=(10, 6))
            # F_f[:, 1] 是 -p56，所以取负变回正值
            plt.scatter(F_f[:, 0], -F_f[:, 1], c='navy', alpha=0.3, label='Pareto Front (Baseline)')
            plt.scatter(final['gwp'], final['p56'], c='red', marker='*', s=250, label='TOPSIS Selected')

            plt.xlabel("Global Warming Potential (GWP)")
            plt.ylabel("56-day Compressive Strength (MPa)")
            plt.title("Baseline Optimization (Numerical Search Only)")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()
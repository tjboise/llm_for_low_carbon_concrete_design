import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
import google.generativeai as genai

# 1. 配置与屏蔽警告
warnings.filterwarnings("ignore")
genai.configure(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
model = genai.GenerativeModel('gemini-2.0-flash')

GWP_FACTORS = {
    'PC': 1.048229, 'FA': 0.328, 'SS': 0.264, 'CAGG': 0.003717, 'FAGG': 0.002576,
    'SF': 0.0, 'WATER': 0.0, 'AEA': 0.0, 'WR_HR': 0.0, 'WR': 0.0, 'ACC': 0.0, 'FIBER': 0.0, 'LATEX': 0.0
}
FEATURES = ['AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC', 'FIBER',
            'LATEX', 'Category']
TARGET = 'fc (MPa)'


class ConcreteBrain:
    def __init__(self, path):
        self.df = pd.read_excel(path).dropna(subset=[TARGET])
        for col in ['PC_TYPE', 'Category']:
            self.df[col] = self.df[col].astype('category').cat.codes
        self.df = self.df.fillna(0)
        self.df['GWP_calc'] = self.df.apply(lambda r: sum(r.get(k, 0) * GWP_FACTORS.get(k, 0) for k in GWP_FACTORS),
                                            axis=1)
        X, y = self.df[FEATURES], self.df[TARGET]
        self.rf = RandomForestRegressor(n_estimators=100).fit(X, y)
        self.xgb = XGBRegressor(n_estimators=100).fit(X, y)
        self.scaler = StandardScaler().fit(X)
        self.mlp = MLPRegressor(hidden_layer_sizes=(64, 64)).fit(self.scaler.transform(X), y)
        self.feature_defaults = self.df[FEATURES].mean().to_dict()

    def predict(self, X_in):
        X_df = pd.DataFrame(X_in)[FEATURES]
        p1, p2 = self.rf.predict(X_df), self.xgb.predict(X_df)
        p3 = self.mlp.predict(self.scaler.transform(X_df))
        return (p1 + p2 + p3) / 3


class ParetoProblem(Problem):
    def __init__(self, brain):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=0, xu=1)
        self.brain = brain
        self.ref_row = pd.DataFrame([brain.feature_defaults])

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        full_x = pd.concat([self.ref_row] * n, ignore_index=True)
        full_x['PC'] = x[:, 0] * 450 + 150
        full_x['FA'] = x[:, 1] * 250
        full_x['WATER'] = x[:, 2] * 120 + 130
        f1 = full_x['PC'] * GWP_FACTORS['PC'] + full_x['FA'] * GWP_FACTORS['FA']
        f2 = -self.brain.predict(full_x)
        out["F"] = np.column_stack([f1, f2])


def main():
    path = 'PA Concrete Database with GWP cement type 5.17.2024.xlsx'
    brain = ConcreteBrain(path)

    # A. 运行 NSGA-II 得到基准
    res = minimize(ParetoProblem(brain), NSGA2(pop_size=100), ('n_gen', 50), seed=1)
    pf = res.F;
    pf[:, 1] = -pf[:, 1]

    # B. 准备 RAG 数据
    top_samples = brain.df[brain.df[TARGET] > 40].sort_values('GWP_calc').head(10)[FEATURES + [TARGET, 'GWP_calc']]

    # C. LLM 设计
    prompt = f"Optimize concrete designs. BEST DATA:\n{top_samples.to_string()}\nREQ: {FEATURES}\nReturn ONLY JSON list with 'name' and 'components' (dict of features)."
    response = model.generate_content(prompt)
    try:
        llm_designs = json.loads(re.search(r'\[.*\]', response.text, re.DOTALL).group(0))
    except:
        return

    # D. 校验
    llm_processed = []
    print("\n" + "=" * 20 + " LLM 设计方案详情 " + "=" * 20)
    for d in llm_designs:
        final_comp = brain.feature_defaults.copy()
        for f in FEATURES:
            if f in d['components']: final_comp[f] = float(d['components'][f])

        s_pred = brain.predict([final_comp])[0]
        g_calc = sum(final_comp[k] * GWP_FACTORS.get(k, 0) for k in GWP_FACTORS)
        llm_processed.append({'g': g_calc, 's': s_pred, 'name': d['name']})
        print(f"方案: {d['name']} | 强度: {s_pred:.2f} MPa | GWP: {g_calc:.2f}")

    # E. 可视化 (不画原始散点)
    plt.figure(figsize=(10, 6))

    # 绘制 Pareto Front (加粗且带有填充)
    idx = np.argsort(pf[:, 0])
    plt.plot(pf[idx, 0], pf[idx, 1], color='#1f77b4', label='Database Pareto Limit (NSGA-II)', lw=3)
    plt.fill_between(pf[idx, 0], pf[idx, 1], color='#1f77b4', alpha=0.1, label='Feasible Design Region')

    # 绘制 LLM 设计点 (增大星号尺寸)
    for pt in llm_processed:
        plt.scatter(pt['g'], pt['s'], c='#d62728', marker='*', s=350, edgecolors='black', linewidths=1.2, zorder=10)
        plt.annotate(pt['name'], (pt['g'] + 4, pt['s'] - 0.5), fontsize=10, fontweight='bold')

    # 图表细节
    plt.title("Benchmarking AI Designs against Historical Pareto Frontier", fontsize=14, pad=15)
    plt.xlabel("Global Warming Potential (lb CO2 eq)", fontsize=12)
    plt.ylabel("fc (MPa) - Compressive Strength", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
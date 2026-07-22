import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from google import genai
import json
import re
import os
import matplotlib.pyplot as plt

# ==========================================================
# 1. API & Schema Definition
# ==========================================================
client = genai.Client(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
MODEL_ID = "gemini-2.0-flash"

# 手动定义 Schema，确保 100% 兼容性
DESIGN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "designs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "components": {
                        "type": "OBJECT",
                        "properties": {
                            "PC": {"type": "NUMBER"}, "PC_TYPE": {"type": "NUMBER"},
                            "FA": {"type": "NUMBER"}, "SS": {"type": "NUMBER"},
                            "SF": {"type": "NUMBER"}, "WATER": {"type": "NUMBER"},
                            "FAGG": {"type": "NUMBER"}, "CAGG": {"type": "NUMBER"},
                            "AEA": {"type": "NUMBER"}, "WR_HR": {"type": "NUMBER"},
                            "WR": {"type": "NUMBER"}, "ACC": {"type": "NUMBER"},
                            "FIBER": {"type": "NUMBER"}, "LATEX": {"type": "NUMBER"}
                        }
                    },
                    "designer_reasoning": {"type": "STRING"}
                },
                "required": ["name", "components", "designer_reasoning"]
            }
        }
    },
    "required": ["designs"]
}

SELECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "best_pick_name": {"type": "STRING"},
        "reasoning": {"type": "STRING"}
    },
    "required": ["best_pick_name", "reasoning"]
}

GWP_FACTORS = {
    'PC': 1.048229, 'FA': 0.328, 'SS': 0.264, 'CAGG': 0.003717, 'FAGG': 0.002576,
    'SF': 0.0, 'WATER': 0.0, 'AEA': 0.0, 'WR_HR': 0.0, 'WR': 0.0, 'ACC': 0.0, 'FIBER': 0.0, 'LATEX': 0.0
}


# ==========================================================
# 2. Optimization Engine
# ==========================================================
class ConcreteMultiModelSystem:
    def __init__(self, data_path):
        self.features = ['AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC',
                         'FIBER', 'LATEX', 'Category']
        print(f"[System] Training Triple-Model Ensemble (RF, XGB, MLP)...")
        self.df = pd.read_excel(data_path, sheet_name='Sheet1').dropna(subset=['fc (MPa)'])
        X = self.df[self.features].fillna(0)
        y = self.df['fc (MPa)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        self.xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        self.scaler = StandardScaler().fit(X_train)
        self.mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42).fit(
            self.scaler.transform(X_train), y_train)
        print(f" -> Models Ready. XGB R2: {r2_score(y_test, self.xgb.predict(X_test)):.4f}")

    def predict_metrics(self, comp):
        vec = {k: 0.0 for k in self.features}
        for k, v in comp.items():
            k_u = k.upper()
            if k_u in vec:
                try:
                    vec[k_u] = float(str(v).replace(',', ''))
                except:
                    pass
        vec['AGE'] = 28
        input_df = pd.DataFrame([vec])[self.features]
        input_scaled = self.scaler.transform(input_df)

        return {
            "RF_Strength": float(self.rf.predict(input_df)[0]),
            "XGB_Strength": float(self.xgb.predict(input_df)[0]),
            "MLP_Strength": float(self.mlp.predict(input_scaled)[0]),
            "LCA_GWP": float(sum(vec[k] * GWP_FACTORS.get(k, 0) for k in GWP_FACTORS)),
            "wb_ratio": float(vec['WATER'] / (vec['PC'] + vec['FA'] + vec['SS'] + vec['SF'] + 0.001)),
            "processed_components": vec
        }


# ==========================================================
# 3. App Pipeline
# ==========================================================
def run_app():
    db_file = 'PA Concrete Database with GWP cement type 5.17.2024.xlsx'
    app = ConcreteMultiModelSystem(db_file)

    target_grade = input("\nEnter Target Grade (e.g. C35, C40): ").upper().strip()
    target_mpa = int(re.search(r'\d+', target_grade).group())
    target_cat = int(input("Enter Category ID (1:Pavement, 4:Bridge): "))

    # STEP 1: Designer
    prompt1 = f"Propose 5 mix designs for {target_grade} (Category {target_cat}). Goal: Minimize GWP using SCMs (FA, SS). Ensure strength >= {target_mpa} MPa."
    print(f"[Designer] Formulating 5 candidate designs...")
    res1 = client.models.generate_content(
        model=MODEL_ID, contents=prompt1,
        config={'response_mime_type': 'application/json', 'response_schema': DESIGN_SCHEMA}
    )
    designs = json.loads(res1.text).get('designs', [])

    # STEP 2: Multi-Model Validator
    print("[Validator] Checking designs against physical truth...")
    validated_list = []
    for d in designs:
        metrics = app.predict_metrics(d['components'])
        # 整合所有信息
        record = {
            "Design_Name": d['name'],
            "Designer_Reasoning": d['designer_reasoning'],
            **metrics['processed_components'],
            "RF_Strength": metrics['RF_Strength'],
            "XGB_Strength": metrics['XGB_Strength'],
            "MLP_Strength": metrics['MLP_Strength'],
            "Mean_Strength": (metrics['RF_Strength'] + metrics['XGB_Strength'] + metrics['MLP_Strength']) / 3,
            "LCA_GWP": metrics['LCA_GWP'],
            "WB_Ratio": metrics['wb_ratio']
        }
        validated_list.append(record)

    # STEP 3: Selector
    # 将结果转换为标准类型以供 Selector 阅读
    def convert(obj):
        if isinstance(obj, list): return [convert(i) for i in obj]
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj

    serializable_list = convert(validated_list)
    prompt2 = f"Review these 5 designs for {target_grade}. Pick the BEST based on GWP minimization and Mean_Strength >= {target_mpa} MPa. Data: {json.dumps(serializable_list)}"

    print("[Selector] Selecting champion and analyzing trade-offs...")
    res2 = client.models.generate_content(
        model=MODEL_ID, contents=prompt2,
        config={'response_mime_type': 'application/json', 'response_schema': SELECTION_SCHEMA}
    )
    selection = json.loads(res2.text)

    # STEP 4: Output & Export
    df_report = pd.DataFrame(validated_list)
    df_report.to_excel("Concrete_Optimization_Report.xlsx", index=False)

    print("\n" + "=" * 70)
    print(f"FINAL DECISION: {selection['best_pick_name']}")
    print(f"SELECTOR REASONING: {selection['reasoning']}")
    print("=" * 70)
    print(f"[System] Full report exported to 'Concrete_Optimization_Report.xlsx'")

    # 可视化前沿图
    plt.figure(figsize=(10, 6))
    plt.scatter(df_report['Mean_Strength'], df_report['LCA_GWP'], c='blue', s=100, label='AI Proposals')
    plt.axvline(x=target_mpa, color='red', linestyle='--', label=f'Target ({target_mpa} MPa)')
    plt.xlabel("Ensemble Mean Strength (MPa)")
    plt.ylabel("GWP (lb CO2 eq)")
    plt.title(f"Pareto Frontier for {target_grade}")
    plt.grid(True, alpha=0.3);
    plt.legend();
    plt.savefig("pareto_frontier.png")


if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor
import google.generativeai as genai
import json
import re
import os
import matplotlib.pyplot as plt

# ==========================================================
# 1. API Configuration & GWP Factors
# ==========================================================
genai.configure(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
model = genai.GenerativeModel('gemini-2.0-flash')

GWP_FACTORS = {
    'PC': 1.048229, 'FA': 0.328, 'SS': 0.264,
    'CAGG': 0.003717, 'FAGG': 0.002576,
    'SF': 0.0, 'WATER': 0.0, 'AEA': 0.0, 'WR_HR': 0.0,
    'WR': 0.0, 'ACC': 0.0, 'FIBER': 0.0, 'LATEX': 0.0
}


# ==========================================================
# 2. Robust Utility Functions
# ==========================================================
def mpa_to_grade(mpa):
    if mpa < 20: return "Below C20"
    for g in range(20, 100, 5):
        if g <= mpa < g + 5: return f"C{g}"
    return "High Performance"


def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    try:
        clean_str = str(val).replace(',', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_str)
        return float(nums[0]) if nums else default
    except:
        return default


def clean_json_string(text):
    """Deep cleans LLM output to fix JSONDecodeErrors."""
    try:
        match = re.search(r'([\[\{].*[\]\}])', text, re.DOTALL)
        if not match: return None
        json_str = match.group(1).replace("'", '"')
        json_str = re.sub(r'(\d),(\d)', r'\1\2', json_str)
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
        json_str = "".join(char for char in json_str if ord(char) >= 32)
        return json_str
    except:
        return None


# ==========================================================
# 3. Optimization Engine
# ==========================================================
class ConcreteMultiModelSystem:
    def __init__(self, data_path):
        self.features = [
            'AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG', 'WATER',
            'AEA', 'WR_HR', 'WR', 'ACC', 'FIBER', 'LATEX', 'Category'
        ]
        print(f"[System] Training Triple-Model Ensemble (RF, XGB, MLP)...")
        self.df = pd.read_excel(data_path, sheet_name='Sheet1').dropna(subset=['fc (MPa)'])
        if 'LATEX' not in self.df.columns: self.df['LATEX'] = 0
        self.df['GWP_calc'] = self.df.apply(lambda r: sum(r.get(k, 0) * GWP_FACTORS.get(k, 0) for k in GWP_FACTORS),
                                            axis=1)

        X = self.df[self.features].fillna(0)
        y = self.df['fc (MPa)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        self.xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
        self.scaler = StandardScaler().fit(X_train)
        self.mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42).fit(
            self.scaler.transform(X_train), y_train)

        self.model_stats = {}
        for name, mdl, is_scaled in [("RF", self.rf, False), ("XGB", self.xgb, False), ("MLP", self.mlp, True)]:
            test_in = self.scaler.transform(X_test) if is_scaled else X_test
            preds = mdl.predict(test_in)
            self.model_stats[name] = {"R2": r2_score(y_test, preds),
                                      "Accuracy": accuracy_score([mpa_to_grade(i) for i in y_test],
                                                                 [mpa_to_grade(i) for i in preds])}
            print(f" -> {name}: R2={self.model_stats[name]['R2']:.4f}, Acc={self.model_stats[name]['Accuracy']:.2%}")

    def retrieve_rag_context(self, target_grade_str, target_cat):
        try:
            min_mpa = int(re.search(r'\d+', target_grade_str).group())
        except:
            min_mpa = 30
        ref = self.df[(self.df['fc (MPa)'] >= min_mpa) & (self.df['Category'] == target_cat)]
        if ref.empty: ref = self.df[self.df['Category'] == target_cat]
        return ref.sort_values(by='GWP_calc').head(8)

    def predict_all_models(self, comp):
        cleaned_vec = {k: safe_float(comp.get(k, 0)) for k in self.features}
        cleaned_vec['AGE'] = 28
        input_df = pd.DataFrame([cleaned_vec])[self.features]
        input_scaled = self.scaler.transform(input_df)
        return {
            "RF_Strength": float(self.rf.predict(input_df)[0]),
            "XGB_Strength": float(self.xgb.predict(input_df)[0]),
            "MLP_Strength": float(self.mlp.predict(input_scaled)[0]),
            "Calculated_GWP": float(sum(cleaned_vec[k] * GWP_FACTORS.get(k, 0) for k in GWP_FACTORS)),
            "Cleaned_Components": cleaned_vec
        }


# ==========================================================
# 4. Final Application Logic
# ==========================================================
def run_app():
    db_file = 'PA Concrete Database with GWP cement type 5.17.2024.xlsx'
    app = ConcreteMultiModelSystem(db_file)

    target_grade_input = input("\nEnter Target Grade (e.g. C35): ").upper().strip()
    target_mpa = int(re.search(r'\d+', target_grade_input).group())
    target_cat = int(input("Enter Category ID (1:Pavement, 4:Bridge): "))

    # STEP 1: RAG
    rag_context = app.retrieve_rag_context(target_grade_input, target_cat)

    # STEP 2: Designer
    designer_prompt = f"Propose 5 designs for {target_grade_input} (Min {target_mpa} MPa). RAG data: {rag_context.to_string()}. Output JSON list with 'name' and 'components'."
    res_designer = model.generate_content(designer_prompt)
    designs_list = json.loads(clean_json_string(res_designer.text))

    # STEP 3: Validator (With Key-Name Defense)
    print("[Validator] Verifying designs...")
    validated_designs = []
    for d in designs_list:
        # Fuzzy search for component keys
        possible_keys = ['components', 'mix', 'ingredients', 'composition', 'proportions']
        comp_key = next((k for k in d.keys() if k.lower() in possible_keys), None)

        # If specific key not found, exclude known non-component keys
        raw_comp = d[comp_key] if comp_key else {k: v for k, v in d.items() if
                                                 k.lower() not in ['name', 'designer_reasoning', 'reasoning']}

        res = app.predict_all_models(raw_comp)
        d.update({
            'RF_Strength': res['RF_Strength'], 'XGB_Strength': res['XGB_Strength'], 'MLP_Strength': res['MLP_Strength'],
            'lca_gwp_val': res['Calculated_GWP'], 'components': res['Cleaned_Components']
        })
        validated_designs.append(d)

    # STEP 4: Selector
    selector_prompt = f"Pick the BEST (Strength >= {target_mpa} and min GWP): {json.dumps(validated_designs, indent=2)}. Output JSON: {{'best_pick_name': '...', 'reasoning': '...'}}"
    res_selector = model.generate_content(selector_prompt)
    val_data = json.loads(clean_json_string(res_selector.text))

    # STEP 5: Visualize & Report
    plt.figure(figsize=(10, 6))
    pdf = pd.DataFrame(validated_designs)
    pdf['Mean_S'] = (pdf['RF_Strength'] + pdf['XGB_Strength'] + pdf['MLP_Strength']) / 3
    plt.scatter(pdf['Mean_S'], pdf['lca_gwp_val'], c='blue', s=100, label='AI Proposals')
    plt.scatter(pdf[pdf['name'] == val_data['best_pick_name']]['Mean_S'],
                pdf[pdf['name'] == val_data['best_pick_name']]['lca_gwp_val'], c='red', marker='*', s=300, label='BEST')
    plt.axvline(x=target_mpa, color='green', linestyle='--')
    plt.xlabel("Mean Predicted Strength (MPa)");
    plt.ylabel("GWP (lb CO2 eq)");
    plt.legend();
    plt.grid(True)
    plt.savefig("optimization_v6_final.png")

    print(f"\nCHAMPION: {val_data['best_pick_name']}\nREASONING: {val_data['reasoning']}")
    pd.DataFrame(validated_designs).to_excel("Final_Optimized_Report.xlsx")


if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
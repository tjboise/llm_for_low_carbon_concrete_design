import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai
import json
import re
import os

# ==========================================================
# 1. 配置 Google AI API (使用 Gemini 2.0 Flash)
# ==========================================================
# 请确保你的 API Key 是有效的
genai.configure(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
model = genai.GenerativeModel('gemini-2.0-flash')


# ==========================================================
# 2. 辅助工具：强度等级映射
# ==========================================================
def mpa_to_grade(mpa):
    """将连续的 MPa 数值映射到离散的 C-Grade 等级"""
    if mpa < 20: return "Below C20"
    if 20 <= mpa < 25: return "C20"
    if 25 <= mpa < 30: return "C25"
    if 30 <= mpa < 35: return "C30"
    if 35 <= mpa < 40: return "C35"
    if 40 <= mpa < 45: return "C40"
    if 45 <= mpa < 50: return "C45"
    if 50 <= mpa < 55: return "C50"
    if 55 <= mpa < 60: return "C55"
    if 60 <= mpa < 70: return "C60"
    if 70 <= mpa < 80: return "C70"
    if 80 <= mpa < 100: return "C80"
    return "C100+"


# ==========================================================
# 3. 物理引擎与 ML 校验器 (Ground-Truth)
# ==========================================================
class PhysicalEngine:
    def __init__(self, data_path):
        # 定义模型需要的 17 个特征
        self.features = [
            'AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG', 'WATER',
            'AEA', 'WR_HR', 'WR', 'ACC', 'FIBER', 'Category', 'w/b', 'SCM%'
        ]
        print(f"[System] Loading data and training Random Forest models...")

        # 加载数据 (Sheet1)
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path, sheet_name='Sheet1')
        else:
            df = pd.read_csv(data_path)

        df = df.dropna(subset=['fc (MPa)', 'GWP'])

        # 训练回归模型
        X = df[self.features]
        self.rf_s = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['fc (MPa)'])
        self.rf_g = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df['GWP'])

        # 记录映射等级
        df['Grade'] = df['fc (MPa)'].apply(mpa_to_grade)
        self.df = df
        print("[System] ML Engine (Ground-Truth) is ready.")

    def get_context(self, grade, cat):
        """为 Designer 获取历史参考数据 (RAG 思想)"""
        ref = self.df[(self.df['Grade'] == grade) & (self.df['Category'] == cat)]
        if ref.empty:
            # 如果没找到完全匹配的，放宽等级限制找最接近的
            ref = self.df[self.df['Category'] == cat].sort_values(by='fc (MPa)')
        return ref.sort_values('GWP').head(5).to_string()

    def verify(self, components):
        """
        利用物理模型进行真理校验。
        包含强大的防御性编程逻辑，处理 LLM 输出的缺失键或字符串。
        """
        # 预设默认值 (防止 LLM 漏掉关键参数)
        defaults = {
            'AGE': 28, 'PC': 0, 'PC_TYPE': 1, 'FA': 0, 'SS': 0, 'SF': 0,
            'FAGG': 0, 'CAGG': 0, 'WATER': 0, 'AEA': 0, 'WR_HR': 0, 'WR': 0,
            'ACC': 0, 'FIBER': 0, 'Category': 4, 'w/b': 0.40, 'SCM%': 0.0
        }

        # 补齐字段
        full_comp = {**defaults, **components}

        # 强制类型转换 (将 LLM 可能输出的字符串如 'CEM I' 拦截并回退到默认值)
        cleaned_comp = {}
        for key in self.features:
            val = full_comp.get(key)
            try:
                cleaned_comp[key] = float(val)
            except (ValueError, TypeError):
                # 如果无法转换为数字，使用 defaults 里的数值
                cleaned_comp[key] = float(defaults.get(key, 0))

        # 构建 DataFrame
        input_df = pd.DataFrame([cleaned_comp])[self.features]

        # 计算真值
        s_val = self.rf_s.predict(input_df)[0]
        g_val = self.rf_g.predict(input_df)[0]
        return s_val, g_val, mpa_to_grade(s_val)


# ==========================================================
# 4. 设计、审计与汇总流水线
# ==========================================================
def run_concrete_pipeline():
    # 数据文件路径
    db_file = 'PA Concrete Database with GWP cement type 5.17.2024.xlsx'
    if not os.path.exists(db_file):
        print(f"[Error] Data file not found at {db_file}")
        return

    # 初始化引擎
    engine = PhysicalEngine(db_file)

    # --- 用户交互输入 ---
    print("\n" + "=" * 60)
    print("LOW-CARBON CONCRETE DESIGN SYSTEM (LLM + ML)")
    print("=" * 60)
    user_grade = input("Target Strength Grade (e.g., C30, C40, C60): ").upper().strip()
    print("Common Categories: 1:Pavement, 4:Bridge Deck, 5:Structural(Exposed), 6:Structural(Free)")
    user_cat = int(input("Select Category ID: "))

    # 获取 RAG 背景知识
    history_context = engine.get_context(user_grade, user_cat)

    # --- LLM 1: Designer ---
    designer_prompt = f"""
    You are the 'Designer'. Propose 3 innovative mix designs for {user_grade}, Category {user_cat}.
    Target: Lowest possible GWP (carbon footprint).

    Reference data (Historical low-GWP mixes for this grade):
    {history_context}

    CRITICAL RULES:
    1. The 'components' dictionary MUST contain all 17 keys as NUMBERS.
    2. PC_TYPE must be an INTEGER (1-13). Do NOT use strings.
    3. If a material is not used, set it to 0.
    Keys: ['AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC', 'FIBER', 'Category', 'w/b', 'SCM%']

    Output strictly in this JSON format:
    [
      {{
        "design_name": "Name",
        "components": {{ ... 17 keys ... }},
        "predicted_strength": float,
        "predicted_gwp": float,
        "predicted_grade": "{user_grade}",
        "reasoning": "English text"
      }},
      ... (total 3 designs)
    ]
    """
    print(f"\n[Designer] Designing 3 proposals for {user_grade}...")
    res1 = model.generate_content(designer_prompt)
    try:
        designs = json.loads(re.search(r'\[.*\]', res1.text, re.DOTALL).group())
    except Exception as e:
        print(f"[Error] Failed to parse Designer's JSON: {e}")
        return

    # --- LLM 2: Validator ---
    validator_prompt = f"""
    You are the 'Validator'. Audit these 3 designs: {json.dumps(designs)}

    Predict the Strength (MPa), Grade (e.g. C30), and GWP based on your expertise.
    Output MUST be a JSON object:
    {{
      "evaluations": [
        {{ "val_strength": float, "val_gwp": float, "val_grade": "string", "val_reasoning": "text" }},
        ...
      ],
      "best_pick": "Design Name",
      "overall_audit_reasoning": "text"
    }}
    """
    print("[Validator] Auditing and predicting performance...")
    res2 = model.generate_content(validator_prompt)
    try:
        val_data = json.loads(re.search(r'\{.*\}', res2.text, re.DOTALL).group())
    except Exception as e:
        print(f"[Error] Failed to parse Validator's JSON: {e}")
        return

    # --- 物理校验与结果导出 ---
    print("\n[System] Performing ML validation and consolidating results...")
    comparison_rows = []
    eval_list = val_data.get('evaluations', [])

    for i, d in enumerate(designs):
        # 运行 ML 物理真理预测
        ml_s, ml_g, ml_grade = engine.verify(d['components'])

        # 获取对应的 Validator 预测
        v = eval_list[i] if i < len(eval_list) else {}

        # 汇总完整行
        row = {
            "Design_Name": d.get('design_name', f"Design_{i + 1}"),
            **d['components'],  # 展开 17 个材料参数
            "Designer_Pred_Strength": d.get('predicted_strength'),
            "Designer_Pred_GWP": d.get('predicted_gwp'),
            "Designer_Pred_Grade": d.get('predicted_grade'),
            "Designer_Reasoning": d.get('reasoning'),
            "Validator_Pred_Strength": v.get('val_strength'),
            "Validator_Pred_GWP": v.get('val_gwp'),
            "Validator_Pred_Grade": v.get('val_grade'),
            "Validator_Reasoning": v.get('val_reasoning'),
            "ML_True_Strength": ml_s,
            "ML_True_GWP": ml_g,
            "ML_True_Grade": ml_grade,
            "Accuracy_Gap_Strength": abs(ml_s - float(v.get('val_strength', 0)))
        }
        comparison_rows.append(row)

    # 保存至 Excel
    result_df = pd.DataFrame(comparison_rows)
    output_filename = "Concrete_Design_Full_Report.xlsx"
    result_df.to_excel(output_filename, index=False)

    # 终端打印总结报告
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION REPORT")
    print("=" * 70)
    print(f"Validator's Choice: {val_data.get('best_pick')}")
    print(f"Audit Reasoning: {val_data.get('overall_audit_reasoning')}")
    print(f"\nExcel Report Saved: {output_filename}")
    print("-" * 70)
    print("ML GROUND-TRUTH SUMMARY:")
    print(result_df[['Design_Name', 'ML_True_Grade', 'ML_True_Strength', 'ML_True_GWP']].to_string(index=False))
    print("=" * 70)


# ==========================================================
# 5. 执行
# ==========================================================
if __name__ == "__main__":
    try:
        run_concrete_pipeline()
    except Exception as e:
        print(f"\n[Final Error] {e}")
import pandas as pd
import numpy as np
import json, re, joblib, warnings
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from google import genai
from neo4j import GraphDatabase

warnings.filterwarnings("ignore")

# --- 1. 配置信息 ---
API_KEY = "AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo"
client = genai.Client(api_key=API_KEY)

# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "Leke123123#"

# GWP 因子
GWP_FACTORS = {'PC': 1.048, 'FA': 0.328, 'SC': 0.264, 'SF': 0.850, 'CAGG': 0.0037, 'FAGG': 0.0026}


# --- 2. 预测大脑：原材料输入 -> 衍生特征计算 -> 链式预测 ---
class XGBChainedBrain:
    def __init__(self):
        try:
            meta = joblib.load('concrete_random_search_chained.pkl')
            self.models = meta['models']
            self.all_feature_names = meta['feature_names']
            self.mins = meta['mins']
            self.maxs = meta['maxs']
            self.independent_vars = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']
        except Exception as e:
            raise FileNotFoundError(f"加载模型失败: {e}")

    def predict_full_workflow(self, raw_mixes_list):
        df = pd.DataFrame(raw_mixes_list)
        df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC'] + df['SF']
        df['w/b'] = df['WATER'] / (df['TOTAL_BINDER'] + 1e-6)
        agg_sum = df['FAGG'] + df['CAGG'] + 1e-6
        df['b/a'] = df['TOTAL_BINDER'] / agg_sum
        df['SCM%'] = (df['FA'] + df['SC'] + df['SF']) / (df['TOTAL_BINDER'] + 1e-6)
        df['CAGG%'] = df['CAGG'] / agg_sum
        df['FAGG%'] = df['FAGG'] / agg_sum

        X_base = df[self.all_feature_names]
        p7 = self.models['7day'].predict(X_base)
        X_28 = X_base.copy();
        X_28['7day'] = p7
        p28 = np.maximum(self.models['28day'].predict(X_28), p7)
        X_56 = X_base.copy();
        X_56['28day'] = p28
        p56 = np.maximum(self.models['56day'].predict(X_56), p28)

        gwp_vals = np.zeros(len(df))
        for mat, factor in GWP_FACTORS.items():
            if mat in df.columns:
                gwp_vals += df[mat].values * factor

        return {'p7': p7, 'p28': p28, 'p56': p56, 'gwp': gwp_vals, 'derivatives': df[['w/b', 'SCM%', 'b/a']]}


# --- 3. NSGA-II 问题定义 ---
class MOOProblem(Problem):
    def __init__(self, brain, targets):
        self.brain, self.targets = brain, targets
        super().__init__(n_var=len(brain.independent_vars), n_obj=2, n_constr=len(targets), xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        rows = []
        for i in range(x.shape[0]):
            d = {v: x[i, j] * (self.brain.maxs[v] - self.brain.mins[v]) + self.brain.mins[v]
                 for j, v in enumerate(self.brain.independent_vars)}
            rows.append(d)

        res = self.brain.predict_full_workflow(rows)
        latest_key = f"p{max([int(k.replace('day', '')) for k in self.targets.keys()])}"
        out["F"] = np.column_stack([res['gwp'], -res[latest_key]])
        out["G"] = np.column_stack([self.targets[age] - res[f"p{age.replace('day', '')}"] for age in self.targets])


# --- 4. 主程序 ---
def main():
    brain = XGBChainedBrain()

    print("\n" + "=" * 50 + "\nLow-carbon concrete design System (Independent Vars Search)\n" + "=" * 50)
    user_query = input("Input your custom requirement: ")

    # --- PROMPT 1: 需求解析 ---
    prompt_goal_parse = f"""
    Instruction: Based on the following user query, extract the concrete performance requirements as an explicit JSON structure suitable for optimization search.

    User Query: {user_query}

    Return JSON keys:
    - "hard_constraints": e.g. {{"fc_28day": {{ "min": 45 }},"fc_56day":{{"min":60}}}} (for day-age property ≥ or ≤)
    - "objectives":  ["min_GWP", "max_fc_28day", "max_fc_56day", ...], sorted with most important first. If user says "minimum carbon" or "balance", infer accordingly.
    - "priority": e.g. "early_strength", "ultra_low_carbon", "balanced", etc. (pick best summary)
    - (optional) "narrative": a plain English summary of what user wants

    Example output:
    {{
      "hard_constraints": {{"fc_28day": {{"min":45}}, "fc_56day":{{"min":60}}}},
      "objectives": ["min_GWP", "max_fc_28day"],
      "priority": "high_strength",
      "narrative": "User wants high ultimate and late strength, with low carbon."
    }}
    """

    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_goal_parse)
    raw_json = json.loads(re.search(r'\{.*\}', resp.text, re.DOTALL).group(0))

    targets = {}
    constraints = raw_json.get("hard_constraints", {})
    for key, val in constraints.items():
        clean_key = key.replace("fc_", "").replace("strength", "").strip()
        if isinstance(val, dict):
            num_val = list(val.values())[0]
            targets[clean_key] = float(num_val)
        else:
            targets[clean_key] = float(val)

    print(f"-> 识别到 NSGA-II 约束目标: {targets}")

    # NSGA-II 搜索
    problem = MOOProblem(brain, targets)
    res_moo = minimize(problem, NSGA2(pop_size=200), ('n_gen', 100), seed=42)

    feasible = (res_moo.G <= 0).all(axis=1)
    if not np.any(feasible):
        print("❌ 未找到可行解");
        return

    # 提取候选集
    candidate_indices = np.argsort(res_moo.F[feasible, 0])[:15]
    candidates = []

    print("\n[Candidate Preview - Top 15 Pareto Solutions]")
    print(f"{'ID':<4} | {'GWP':<8} | {'Strength':<10} | {'w/b':<6} | {'SCM%':<6}")

    for i, idx in enumerate(candidate_indices):
        real_idx = np.where(feasible)[0][idx]
        x_raw = res_moo.X[real_idx]
        recipe = {v: float(x_raw[j] * (brain.maxs[v] - brain.mins[v]) + brain.mins[v]) for j, v in
                  enumerate(brain.independent_vars)}

        res_single = brain.predict_full_workflow([recipe])
        recipe.update({
            'GWP': float(res_moo.F[real_idx, 0]),
            'fc_target': float(-res_moo.F[real_idx, 1]),
            'w/b': float(res_single['derivatives']['w/b'][0]),
            'SCM%': float(res_single['derivatives']['SCM%'][0]),
            'id': int(i)  # 使用 0-14 简化 ID 供 LLM 识别
        })
        candidates.append(recipe)
        print(
            f"{i:<4} | {recipe['GWP']:<8.2f} | {recipe['fc_target']:<10.2f} | {recipe['w/b']:<6.3f} | {recipe['SCM%']:<6.2%}")

    # LLM + lb 决策
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    with driver.session() as session:
        rules = [r["rule"] for r in session.run("MATCH (a)-[r]->(b) RETURN a.name+' '+type(r)+' '+b.name AS rule")]

    # --- PROMPT 2: 专家决策 ---
    prompt_select = f"""
    You are an AI concrete mix designer using multi-objective optimization. You have:
    - A list of candidate mixes as Pareto-optimal solutions (minimizing GWP, maximizing strength). Each candidate contains: all variable values, 'w/b', 'SCM%', 'GWP', 'fc_target', 'id', etc.
    - A set of domain knowledge rules from a knowledge graph (concrete technology best practices, constraints).

    Step 1. For each candidate, check whether it violates any key physical constraints (especially 'w/b' ratio >0.24, 'SCM%' within reasonable range, etc.) given the rules: {rules}.
    Step 2. Among physically reasonable candidates, select the one *closest to the GWP minimum* (i.e., on the Pareto front), unless that would significantly risk property or feasibility.
    Step 3. If the lowest-GWP candidate is not fully physically compliant, recommend a nearby more feasible candidate—explain what the trade-off is.
    Step 4. For your chosen candidate, provide:
        - Which point on the Pareto front it is (e.g., does it have the absolute minimum GWP? or a compromise design slightly higher GWP for better physics?)
        - Your reasoning: trade-offs, knowledge rule checks, why this design is optimal.
        - If no candidate is physically feasible, explain why (e.g., all violate w/b or SCM%).

    Input - Rules from lb: {rules}
    Input - Pareto candidates: {json.dumps(candidates, indent=2)}

    OUTPUT (JSON):
    {{
        "best_id": ...,
        "reasoning": "...",
        "on_pareto_front_position": "absolute_min_gwp" | "compromise_for_physical_feasibility",
        "violated_rules": [ ... ] 
    }}
    """

    resp_select = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_select)
    decision = json.loads(re.search(r'\{.*\}', resp_select.text, re.DOTALL).group(0))

    # --- 结果展示与绘图 ---
    best_mix = next(item for item in candidates if item['id'] == decision['best_id'])

    plt.figure(figsize=(10, 6))
    plt.scatter(res_moo.F[feasible, 0], -res_moo.F[feasible, 1], c='lightgray', alpha=0.5)
    plt.scatter(best_mix['GWP'], best_mix['fc_target'], c='red', s=150, edgecolors='black', label='LLM Selected',
                zorder=5)
    plt.xlabel('GWP (lb CO2e/yd3)');
    plt.ylabel('Strength (MPa)');
    plt.legend();
    plt.show()

    print("\n" + "*" * 20 + " FINAL RECOMMENDED MIX RECIPE " + "*" * 20)
    print(f"Strategy: {decision['on_pareto_front_position']}")
    print(f"Expert Reasoning: {decision['reasoning']}")
    print("-" * 50)
    print(f"{'Material Component':<20} | {'Usage (lb/m³)':<15}")
    print("-" * 50)
    for v in brain.independent_vars:
        print(f"{v:<20} | {best_mix[v]:>15.2f}")
    print("-" * 50)
    print(f"{'Derived Metric':<20} | {'Value':<15}")
    print(f"{'w/b ratio':<20} | {best_mix['w/b']:>15.3f}")
    print(f"{'SCM %':<20} | {best_mix['SCM%']:>15.2%}")
    print(f"{'GWP (Carbon)':<20} | {best_mix['GWP']:>15.2f}")
    print(f"{'Target Strength':<20} | {best_mix['fc_target']:>15.2f}")
    print("*" * 50)

    driver.close()


if __name__ == "__main__":
    main()
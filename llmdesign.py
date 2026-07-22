import pandas as pd
import numpy as np
import json, re, joblib, warnings
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from google import genai
from neo4j import GraphDatabase
import catboost as cb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import json, re

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# --- 1. 基础配置 ---
API_KEY = "AIzaSyDnV_LdQ2aztxCjwuEckEFFYQfc-se4ERA"
client = genai.Client(api_key=API_KEY)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "Leke123123#"

# 碳排放因子 (lb CO2e/lb)
GWP_FACTORS = {'PC': 1.048, 'FA': 0.328, 'SC': 0.264, 'SF': 0.850, 'CAGG': 0.0037, 'FAGG': 0.0026}





# --- 1. RAG 辅助函数 ---
def render_ctx(docs):
    context_parts = []
    for i, doc in enumerate(docs):
        # 这里的 metadata 取法取决于你的 PyPDFDirectoryLoader 配置
        # 通常 PyPDFLoader 会把路径存为 'source'
        source = doc.metadata.get('source', 'Standard_Doc').split('/')[-1] # 只取文件名
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content.strip()[:1000] # 每段强制限制 1000 字符
        context_parts.append(f"SOURCE_{i+1}: [{source} | Page {page}]\nCONTENT: {content}")
    return "\n\n".join(context_parts)

def summarize_candidates_for_query(candidates, user_query):
    """提取候选集摘要，帮助 Query Builder 找准检索方向"""
    # 重点提取极端值和潜在违规点
    avg_wb = sum(c['w/b'] for c in candidates) / len(candidates)
    max_scm = max(c['SCM%'] for c in candidates)
    return {
        "user_intent": user_query,
        "avg_wb": round(avg_wb, 3),
        "max_scm_ratio": round(max_scm, 3),
        "target_strengths": "extracted from query", # solve 里的 targets
        "sample_candidate": {k: v for k, v in candidates[0].items() if k in ['PC', 'w/b', 'SCM%', 'AEA']}
    }

# --- 初始化向量数据库 ---

from langchain_community.embeddings import HuggingFaceEmbeddings
def initialize_rag_db(api_key):
    index_path = "faiss_index"
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="text-embedding-004",
    #     google_api_key=api_key,
    #     task_type="retrieval_document"
    # )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 如果本地已经有索引，直接加载
    if os.path.exists(index_path):
        print("Detect local index, loading...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("No detecting local index, loading PDF and build vector...")
        loader = PyPDFDirectoryLoader("standard/")
        docs = loader.load()
        if not docs:
            raise FileNotFoundError("error：'standard/' no PDF files。")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(splits, embeddings)
        # 将生成的索引保存到本地
        vectorstore.save_local(index_path)
        print("local index save to faiss_index")

    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 2. 预测大脑：执行物理特征推导与链式预测 ---
class CatBoostChainedBrain:
    def __init__(self):
        try:
            # 1. 加载我们刚训练好的 CatBoost 序列化文件
            meta = joblib.load('concrete_catboost_optimized.pkl')
            self.models = meta['models']
            self.all_feature_names = meta['feature_names']
            # 这里需要注意，CatBoost 内部可能不需要 mins/maxs 归一化，
            # 但为了兼容 NSGA-II 的 [0,1] 搜索空间，我们依然保留这些元数据。
            # 如果你之前保存的 meta 没变，可以继续用。
            self.mins = meta.get('mins', {})
            self.maxs = meta.get('maxs', {})
            self.independent_vars = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']

            print("analyzing train data statistical features...")
            df_train = pd.read_csv('Super_Cleaned_Concrete_Data.csv')

            # 定义计算公式 (和你 predict 里的逻辑保持一致)
            df_train['TOTAL_BINDER'] = df_train['PC'] + df_train['FA'] + df_train['SC'] + df_train['SF']
            df_train['w/b'] = df_train['WATER'] / (df_train['TOTAL_BINDER'].replace(0, np.nan))
            agg_sum = df_train['FAGG'] + df_train['CAGG']
            df_train['b/a'] = df_train['TOTAL_BINDER'] / (agg_sum.replace(0, np.nan))
            df_train['SCM%'] = (df_train['FA'] + df_train['SC'] + df_train['SF']) / (
                df_train['TOTAL_BINDER'].replace(0, np.nan))

            # 提取边界
            stats_features = ['PC', 'WATER', 'TOTAL_BINDER', 'w/b', 'b/a', 'SCM%', 'WR_HR', 'ACC']
            self.data_bounds = {}
            for feat in stats_features:
                if feat in df_train.columns:
                    self.data_bounds[feat] = {
                        "min": round(float(df_train[feat].min()), 3),
                        "max": round(float(df_train[feat].max()), 3)
                    }
            print("Analyzing finished!")

        except Exception as e:
            raise FileNotFoundError(f"loading CatBoost fialed: {e}")

    def predict_full_workflow(self, raw_mixes_list):
        df = pd.DataFrame(raw_mixes_list)

        # --- 实时计算基础衍生特征 ---
        df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC'] + df['SF']
        df['w/b'] = df['WATER'] / (df['TOTAL_BINDER'])
        agg_sum = df['FAGG'] + df['CAGG']
        df['b/a'] = df['TOTAL_BINDER'] / (agg_sum)
        df['SCM%'] = (df['FA'] + df['SC'] + df['SF']) / (df['TOTAL_BINDER'])
        df['CAGG%'] = df['CAGG'] / (agg_sum)
        df['FAGG%'] = df['FAGG'] / (agg_sum)
        df['PC%'] = df['PC'] / (df['TOTAL_BINDER'])
        df['FA%'] = df['FA'] / (df['TOTAL_BINDER'])
        df['SC%'] = df['SC'] / (df['TOTAL_BINDER'])

        # 确保输入模型的特征列顺序与训练时完全一致
        # X_base 必须包含上面新增的 4 个特征
        X_base = df[self.all_feature_names]

        # --- 链式预测逻辑 (CatBoost 版) ---
        p7 = self.models['7day'].predict(X_base)

        # 预测 28天：加入 7d 预测值作为输入
        X_28 = X_base.copy()
        X_28['7day'] = p7
        p28 = self.models['28day'].predict(X_28)
        # 物理修正：28天强度不应低于 7天
        p28 = np.maximum(p28, p7)

        # 预测 56天：加入 28d 预测值作为输入
        X_56 = X_base.copy()
        X_56['28day'] = p28
        p56 = self.models['56day'].predict(X_56)
        # 物理修正：56天强度不应低于 28天
        p56 = np.maximum(p56, p28)

        # 计算 GWP (保持不变)
        gwp_vals = np.zeros(len(df))
        for mat, factor in GWP_FACTORS.items():
            if mat in df.columns:
                gwp_vals += df[mat].values * factor

        return {'p7': p7, 'p28': p28, 'p56': p56, 'gwp': gwp_vals, 'df_all': df}

    def predict_full_workflow_for_moo(self, x):
        """专门为 pymoo 优化器准备的接口：将 [0,1] 空间转回物理空间并预测"""
        # 1. 将 pymoo 的 [0,1] 矩阵转为物理单位的列表
        raw_mixes = []
        for row in x:
            recipe = {}
            for j, v in enumerate(self.independent_vars):
                # 使用保存的 mins/maxs 进行逆归一化
                recipe[v] = float(row[j] * (self.maxs[v] - self.mins[v]) + self.mins[v])
            raw_mixes.append(recipe)

        # 2. 调用原有的预测逻辑
        perf = self.predict_full_workflow(raw_mixes)

        # 3. 返回一个方便字典取值的格式
        return {
            'p7': perf['p7'],
            'p28': perf['p28'],
            'p56': perf['p56'],
            'gwp': perf['gwp']
        }

    def get_physical_val(self, x, feature_name):
        """辅助函数：从 [0,1] 矩阵中提取特定特征的物理值，用于计算约束 G"""
        if feature_name in self.independent_vars:
            idx = self.independent_vars.index(feature_name)
            return x[:, idx] * (self.maxs[feature_name] - self.mins[feature_name]) + self.mins[feature_name]

        # 如果是计算出来的衍生变量（如 w/b），则需要重新计算
        # 注意：这里假设 x 是当前种群的矩阵
        raw_mixes = []
        for row in x:
            recipe = {v: row[j] * (self.maxs[v] - self.mins[v]) + self.mins[v]
                      for j, v in enumerate(self.independent_vars)}
            raw_mixes.append(recipe)

        df = pd.DataFrame(raw_mixes)
        if feature_name == 'w/b':
            return (df['WATER'] / (df['PC'] + df['FA'] + df['SC'] + df['SF'])).values
        if feature_name == 'SCM%':
            return ((df['FA'] + df['SC'] + df['SF']) / (df['PC'] + df['FA'] + df['SC'] + df['SF'])).values
        if feature_name == 'TOTAL_BINDER':
            return (df['PC'] + df['FA'] + df['SC'] + df['SF']).values

            # 2. 骨胶比 (Binder to Aggregate ratio)
        if feature_name == 'b/a':
            agg_sum = df['FAGG'] + df['CAGG']
            return ((df['PC'] + df['FA'] + df['SC'] + df['SF']) / agg_sum.replace(0, np.nan)).values

            # 3. 各种成分的占比 (如果审计员说：粉煤灰不能超过胶凝材料的 30%)
        if feature_name == 'FA%':
            return (df['FA'] / (df['PC'] + df['FA'] + df['SC'] + df['SF'])).values
        if feature_name == 'SC%':
            return (df['SC'] / (df['PC'] + df['FA'] + df['SC'] + df['SF'])).values
        return np.zeros(len(x))


# --- 3. 支持动态注入约束的 NSGA-II 问题类 ---
class IterativeMOOProblem(Problem):
    def __init__(self, brain, targets, dynamic_constraints, objectives_list):
        self.brain = brain
        self.targets = targets
        self.dyn_cons = dynamic_constraints
        self.objectives_list = objectives_list

        # 核心修复 1：动态计算约束数量（强度目标数 + LLM 注入的规则数）
        n_constr = len(targets) + len(dynamic_constraints)

        # 核心修复 2：动态计算目标数量
        n_obj = len(objectives_list)

        super().__init__(
            n_var=len(brain.independent_vars),
            n_obj=n_obj,
            n_constr=n_constr,
            xl=0, xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 将 [0,1] 空间转回物理量进行预测
        res = self.brain.predict_full_workflow_for_moo(x)

        # --- 动态处理 F (Objectives) ---
        f_list = []
        for obj in self.objectives_list:
            if obj == "min_GWP":
                f_list.append(res['gwp'])
            elif obj == "max_fc_7day":
                f_list.append(-res['p7'])
            elif obj == "max_fc_28day":
                f_list.append(-res['p28'])
            elif obj == "max_fc_56day":
                f_list.append(-res['p56'])
        out["F"] = np.column_stack(f_list)

        # --- 动态处理 G (Constraints) ---
        g_list = []
        # 1. 强度硬约束 (targets)
        for day_key, min_val in self.targets.items():
            pred_val = res[f'p{day_key.replace("day", "")}']
            g_list.append(min_val - pred_val)

        # 2. LLM 动态注入约束 (dyn_cons)
        for con in self.dyn_cons:
            # 假设你的 brain 有一个方法根据特征名从 x 中提取当前物理值
            curr_val = self.brain.get_physical_val(x, con['feature'])
            if con['op'] == "<=":
                g_list.append(curr_val - con['val'])
            elif con['op'] == ">=":
                g_list.append(con['val'] - curr_val)

        out["G"] = np.column_stack(g_list)




# --- 4. 迭代管理器：协调 LLM 审计与算法搜索 ---
class ConcreteAIManager:
    def __init__(self, brain, retriever):
        self.brain = brain
        self.retriever = retriever #RAG
        self.dynamic_constraints = []
        self.last_user_query = ""

    def audit_candidates(self, candidates, kg_rules, data_bounds):
        """
        落地你设计的三段式 RAG 审计管线
        """
        # --- Step 1: Query Builder ---
        brief = summarize_candidates_for_query(candidates, self.last_user_query)
        qb_prompt = f"""
                You are an expert retrieval query writer for concrete engineering standards (ACI, ASTM).
                Your goal is to generate search queries that will find the specific clauses needed to audit the current mix candidates.

                User intent: {self.last_user_query}
                Candidate summary: {json.dumps(brief)}
                Already applied rules: {json.dumps(self.dynamic_constraints)}

                ### FEW-SHOT EXAMPLES FOR QUERY GENERATION:

                Example 1 (High w/b detected):
                - Candidate Info: "avg_wb": 0.85, "user_intent": "high strength bridge deck"
                - Output JSON: 
                [
                    "ACI 318-19 Table 19.3.2.1 maximum permissible w/c ratio",
                    "ACI 318-19 exposure categories for bridge decks",
                    "ACI 211.1 recommended water-cement ratio for high strength concrete",
                    "relationship between water-cement ratio and compressive strength Abrams Law"
                ]

                Example 2 (High SCM replacement detected):
                - Candidate Info: "max_scm_ratio": 0.65, "user_intent": "low carbon foundation"
                - Output JSON:
                [
                    "ASTM C595 allowable replacement limits for fly ash and slag",
                    "ACI 318-19 limitations on SCM for durability",
                    "impact of high SCM replacement on concrete setting time and strength"
                ]

                ### TASK:
                Based on the provided candidate summary and user intent, return a JSON array of 6-8 concise, technical search queries.
                Focus on: durability, w/c or w/b limits, exposure classes, SCM replacement limits, and admixture dosages.

                Return a JSON array of STRINGS ONLY. Do NOT return objects.
                JSON ONLY.
                """
        qb_resp = client.models.generate_content(model="gemini-2.0-flash", contents=qb_prompt)
        queries = json.loads(re.search(r'\[.*\]', qb_resp.text, re.DOTALL).group(0))

        # --- Step 2: Context Retrieval ---
        ctx_passages = []
        for q in queries:
            # 核心修复：如果 q 是字典（例如 {"query": "..."}），提取出字符串
            if isinstance(q, dict):
                # 尝试常见的 key，或者取第一个值
                search_str = q.get("query") or q.get("search_query") or list(q.values())[0]
            else:
                search_str = str(q)

            # 确保 search_str 不是空的
            if search_str:
                ctx_passages.extend(self.retriever.invoke(search_str))

        # 去重并渲染（保留前15个最相关的段落以防 context 过长）
        context_text = render_ctx(ctx_passages[:15])

        # --- Step 3: Standards Grounded Auditor ---
        auditor_prompt = f"""
                You are a Senior Concrete Materials Expert performing a standards-grounded audit.

                [KNOWLEDGE BASE A: AUTHORITATIVE STANDARDS (RAG)]
                {context_text}

                [KNOWLEDGE BASE B: EMPIRICAL DATA BOUNDS (Training History)]
                Experimental min/max ranges. Use this to detect AI model extrapolation errors:
                {json.dumps(data_bounds, indent=2)}

                [CANDIDATES TO EVALUATE]
                {json.dumps(candidates, ensure_ascii=False)}

                [ALLOWED FEATURES FOR CONSTRAINTS]
                ["PC","FA","SC","SF","FAGG","CAGG","WATER","AEA","WR_HR","WR","ACC","TOTAL_BINDER","w/b","b/a","SCM%","CAGG%","FAGG%","PC%","FA%","SC%"]

                [CORE AUDIT RULES]
                1. REGULATORY PRIMACY: If the [KNOWLEDGE BASE A] provides a specific limit (e.g., ACI 318 Table 19.3.2.1 max w/c), this is the HIGHEST authority. 
                2. ABRAMS' LAW CHECK: A water-to-binder (w/b) ratio > 0.6 is physically incompatible with strengths > 30 MPa. This is a non-negotiable physical limit.
                3. DATA BOUNDS CHECK: If a candidate value exceeds [KNOWLEDGE BASE B] max/min, the AI model prediction is UNRELIABLE.

                ### FEW-SHOT EXAMPLES FOR AUDIT LOGIC:

                Example 1 (Rejected based on ACI 318 Specification):
                - Context: "SOURCE_2: [ACI 318-19 | Page 102] CONTENT: Table 19.3.2.1: Max w/c for Class F1 exposure is 0.45."
                - Candidate: {{"id": 0, "w/b": 0.52, "p28": 35.0}}
                - Output JSON Object:
                {{
                    "id": 0,
                    "status": "REJECTED",
                    "reasoning": "REJECTED per SOURCE_2 (ACI 318-19 p102): 'Max w/c for Class F1 exposure is 0.45'. The design w/b (0.52) violates this durability requirement.",
                    "new_constraints": [
                        {{"feature": "w/b", "op": "<=", "val": 0.45, "description": "Mandatory w/b limit per ACI 318 Table 19.3.2.1", "source": "SOURCE_2:ACI 318-19"}}
                    ]
                }}

                Example 2 (Rejected based on Empirical Bounds & Physical Laws):
                - Empirical Bounds: {{"w/b": {{"min": 0.3, "max": 0.714}}}}
                - Candidate: {{"id": 3, "w/b": 0.85, "p28": 45.0}}
                - Output JSON Object:
                {{
                    "id": 3,
                    "status": "REJECTED",
                    "reasoning": "REJECTED based on Data Bounds and Abrams' Law. 1) The w/b (0.85) far exceeds the experimental maximum of 0.714. 2) Physically unrealistic: per Abrams' Law, w/b 0.85 cannot achieve 45 MPa.",
                    "new_constraints": [
                        {{"feature": "w/b", "op": "<=", "val": 0.45, "description": "Correcting model extrapolation error", "source": "Empirical Data Bounds"}}
                    ]
                }}
                
                Example 3 (Within Data Bounds but Violates Industry Standards):
                - Context: "SOURCE_3: [ACI 318-19 | Page 105] CONTENT: For Exposure Class C2 (Severe Corrosion), max w/c ratio is 0.40 and min strength is 35 MPa."
                - Empirical Bounds: {{"w/b": {{"min": 0.3, "max": 0.714}}}}
                - Candidate: {{"id": 5, "w/b": 0.55, "p28": 38.0}}
                - Output JSON Object:
                {{
                    "id": 5,
                    "status": "REJECTED",
                    "reasoning": "REJECTED per SOURCE_3 (ACI 318 p105). Although the w/b (0.55) is within historical Data Bounds (max 0.714), it violates the strict 0.40 limit required for Exposure Class C2. For bridge decks in severe environments, durability standards override experimental maximums.",
                    "new_constraints": [
                        {{"feature": "w/b", "op": "<=", "val": 0.40, "description": "Mandatory w/b limit for C2 durability compliance", "source": "SOURCE_3:ACI 318-19 Table 19.3.2.1"}}
                    ]
                }}

                ### MANDATORY INSTRUCTIONS:
                1. CITATION REQUIREMENT: For EVERY decision, you MUST cite the SOURCE_ID from Knowledge Base A (e.g., SOURCE_1: Page X) AND/OR reference Knowledge Base B.
                2. QUOTATION REQUIREMENT: You MUST quote the exact sentence from the PDF context if it defines a limit.
                3. DUAL-VALIDATION: Check against BOTH the standards (RAG) and the experimental ranges (Data Bounds).
                4. UNIT CHECK: All strength values (p7, p28, p56) are in MPa.
                5. Return a JSON ARRAY of audit objects. JSON ONLY.
                """
        aud_resp = client.models.generate_content(model="gemini-2.0-flash", contents=auditor_prompt)


        per_cand_results = json.loads(re.search(r'\[.*\]', aud_resp.text, re.DOTALL).group(0))

        # --- 在 Step 3 (aud_resp) 之后添加 ---
        print("\n🔍 [DEBUG] Step 3: Auditor Original Output (Per Candidate):")
        # 尝试美化打印，如果解析失败则打印原文本
        try:
            print(json.dumps(per_cand_results, indent=2, ensure_ascii=False))
        except:
            print(aud_resp.text)



        # --- Step 4: Normalizer ---
        normalizer_prompt = f"""
        Summarize these audit results:
        {json.dumps(per_cand_results, ensure_ascii=False)}
        Instructions: 
        1. Return a single JSON object with "status", "best_id", "reasoning", and a deduplicated "new_constraints" list.
        2. Pick the strictest value if ranges overlap. 
        3. If all candidates are reasonable and no new constraints are needed, you should return status: "APPROVED". Do not keep searching if the goal is met.
        4. From the 15 candidates, pick the best_id that best satisfies the user's priority (e.g., if priority is early_strength, pick the one with highest p7).
        5. Your 'reasoning' MUST include the specific quotes and source citations (e.g., SOURCE_X) provided by the auditor. 
        6. Do NOT just say 'it meets standards'. Say 'It meets standards because [Exact Quote] from [Source Name]'.


        Example 1 (Multiple conflicting limits with RAG evidence):
            - Input Audits: 
                * ID 0: REJECTED. "w/b 0.6 exceeds SOURCE_1 (ACI 318 p102): 'Max w/c for F1 exposure is 0.50'."
                * ID 1: REJECTED. "w/b 0.5 exceeds SOURCE_2 (ACI 318 p105): 'Max w/c for F3 exposure is 0.45'."
            - Output JSON:
            {{
                "status": "REJECTED",
                "new_constraints": [
                    {{"feature": "w/b", "op": "<=", "val": 0.45}} 
                ],
                "reasoning": "Standardized w/b limit to 0.45 based on the strictest requirement. Reference SOURCE_2 (ACI 318 p105): 'Max w/c for F3 exposure is 0.45'. This is stricter than the 0.50 limit in SOURCE_1."
            }}

        Example 2 (Strength vs. Unit conversion):
        - Input Audits:
            * ID 5: REJECTED. "28d strength 30 MPa is below the 35 MPa requirement for this project."
        - Output JSON:
        {{
            "status": "REJECTED",
            "new_constraints": [
                {{"feature": "p28", "op": ">=", "val": 35}}
            ],
            "reasoning": "Increased minimum 28d strength requirement to 35 MPa per project specifications."
        }}

        Example 3 (All Accepted with evidence):
            - Input Audits:
                * ID 0-14: ACCEPTED. "Meets ACI 318. Reference SOURCE_3: 'Concrete shall have a minimum strength of 35MPa for this application'."
            - Output JSON:
            {{
                "status": "APPROVED",
                "best_id": 0,
                "new_constraints": [],
                "reasoning": "All candidates satisfy standards. ID 0 selected as optimal GWP. Confirmed by SOURCE_3: 'Concrete shall have a minimum strength of 35MPa'. Predicted strengths and w/b ratios are within both ACI limits and Empirical Data Bounds."
            }}
        """
        norm_resp = client.models.generate_content(model="gemini-2.0-flash", contents=normalizer_prompt)
        audit = json.loads(re.search(r'\{.*\}', norm_resp.text, re.DOTALL).group(0))

        # --- 在 Step 4 (norm_resp) 之后添加 ---
        print("\n🔍 [DEBUG] Step 4: Normalizer Combined Output (Summary):")
        try:
            print(json.dumps(audit, indent=2, ensure_ascii=False))
        except:
            print(norm_resp.text)

        # --- Step 5: Final Mapping & Feature Cleaning (原有逻辑) ---
        for con in audit.get("new_constraints", []):
            feature_map = {"Total Binder": "TOTAL_BINDER", "Water/Binder": "w/b", "SCM ratio": "SCM%"}
            con["feature"] = feature_map.get(con["feature"], con["feature"])

        # --- 核心修改：合并 Normalizer 结果与调试信息 ---
        return {
            **audit,  # 这是 Normalizer 汇总后的 status, best_id, new_constraints
            "debug_queries": queries,  # Step 1 产生的检索词
            "debug_context_text": context_text  # Step 2 检索到的 PDF 原文
        }

    def _print_candidate_table(self, candidates, iteration):
        import pandas as pd
        # 将列表转换为 DataFrame
        df_view = pd.DataFrame(candidates)

        # 只选择关键展示列，并重命名以方便阅读
        display_cols = {
            'id': 'ID',
            'GWP': 'GWP',
            'w/b': 'w/b',
            'SCM%': 'SCM%',
            'p7': '7d_MPa',
            'p28': '28d_MPa',
            'p56': '56d_MPa'
        }

        # 过滤并排序
        df_display = df_view[list(display_cols.keys())].rename(columns=display_cols)

        # 格式化百分比和小数点
        df_display['SCM%'] = df_display['SCM%'].map(lambda x: f"{x:.1%}")
        for col in ['GWP', '7d_MPa', '28d_MPa', '56d_MPa']:
            df_display[col] = df_display[col].map(lambda x: f"{x:.2f}")
        df_display['w/b'] = df_display['w/b'].map(lambda x: f"{x:.3f}")

        print(f"\n📊 [Iteration {iteration}] Picked Designs (GWP rank from low to high):")
        # 使用 to_string() 打印完整表格，不显示索引
        print(df_display.to_string(index=False))
        print("-" * 80)


    def solve(self, user_query, max_iters=10):
        # 第一步：解析用户自然语言需求
        self.last_user_query = user_query  # 必须添加这一行！

        prompt_parse = f"""
        Instruction: Based on the following user query, extract the concrete performance requirements. 
        You must translate qualitative needs (e.g., "high strength", "low carbon") into specific numeric search goals and objectives.
        
        RULES:
        1. If the user specifies a class like "C2", keep it in 'hard_constraints' as a string.
        2. If the user asks for "high early strength", set a high target for 7d.
        3. If the user asks for "high late-age strength", set a high target for 56d.
        4. If the user provides a single strength (e.g., "50MPa"), assume it's for 28d, then estimate reasonable 7d (approx 70% of 28d) and 56d (approx 110% of 28d) targets to guide the multi-objective optimization.
        
        LOGIC:
        1. Default objective is ALWAYS ["min_GWP"]. 
        2. ONLY add "max_fc_xday" to 'objectives' if the user uses words like "maximize", "as high as possible", or "highest strength".
        3. If the user says "at least" or "minimum", put it in 'hard_constraints' only.


        User Query: {user_query}

        ### FEW-SHOT EXAMPLES FOR DEMAND PARSING:

        Example 1:
        - User: "I need concrete with at least 30 MPa at 28 days. Minimize the carbon footprint."
        - Output JSON:
        {{
          "hard_constraints": {{
              "fc_28day": {{"min": 30}}
          }},
          "objectives": ["min_GWP"],
          "priority": "low_carbon",
          "narrative": "Standard 30MPa requirement. User explicitly asked to minimize GWP only. No strength maximization objective added."
        }}

        Example 2:
        - User: "Strength must be over 35 MPa, but I want the highest possible early strength for rapid construction, while staying eco-friendly."
        - Output JSON:
        {{
          "hard_constraints": {{
              "fc_28day": {{"min": 35}}
          }},
          "objectives": ["min_GWP", "max_fc_7day"],
          "priority": "early_strength",
          "narrative": "35MPa is the floor. User wants to actively push for higher 7d strength and lower GWP."
        }}

        Example 3:
        - User: "Maximize 28d strength and minimize carbon. No specific minimums."
        - Output JSON:
        {{
          "hard_constraints": {{}},
          "objectives": ["min_GWP", "max_fc_28day"],
          "priority": "performance_balanced",
          "narrative": "No hard floor set. Multi-objective optimization to find the best trade-off between GWP and 28d strength."
        }}

        Example 4:
        - User: "Design a mix for a sustainable foundation. Needs to be very low carbon. Strength should be typical for residential use, around 3000 to 4000 psi."
        - Output JSON:
        {{
          "hard_constraints": {{
              "fc_28day": {{"min": 21, "max": 28}}
          }},
          "objectives": ["min_GWP", "max_fc_28day"],
          "priority": "ultra_low_carbon",
          "narrative": "Residential foundation (21-28 MPa equivalent to 3000-4000 psi) prioritized for minimum GWP."
        }}

        Example 5:
        - User: "28d strength must be 40-50 MPa. 56d should be over 60 MPa. Minimize cement content."
        - Output JSON:
        {{
          "hard_constraints": {{
              "fc_28day": {{"min": 40, "max": 50}},
              "fc_56day": {{"min": 60}}
          }},
          "objectives": ["min_GWP", "max_fc_56day"],
          "priority": "balanced",
          "narrative": "Specific strength window for 28d and a minimum for 56d, with cement reduction as the main driver."
        }}

        ### TASK:
        Analyze the user's query and output a JSON object with the keys: 
        "hard_constraints", "objectives", "priority", and "narrative".
        Translate psi to MPa if necessary (1000 psi ≈ 6.9 MPa).
        Return JSON ONLY.
        """


        resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_parse)
        raw_req = json.loads(re.search(r'\{.*\}', resp.text, re.DOTALL).group(0))

        print("\n" + "=" * 20 + " Demand Analysis Report " + "=" * 20)
        print(f"User Narrative: {raw_req.get('narrative', 'N/A')}")
        print(f"Optimization Priority: {raw_req.get('priority', 'N/A')}")
        print("\n[Translated Optimization Targets]:")

        # 遍历解析出的硬性约束并打印
        constraints = raw_req.get("hard_constraints", {})
        for feat, limit in constraints.items():
            if isinstance(limit, dict):
                limit_str = f"Min: {limit.get('min', 'N/A')}, Max: {limit.get('max', 'N/A')}"
            else:
                limit_str = str(limit)
            print(f" - {feat}: {limit_str}")

        print(f"Objectives: {', '.join(raw_req.get('objectives', []))}")
        print("=" * 60 + "\n")

        # 转换为扁平的 targets 字典
        targets = {}
        for k, v in constraints.items():
            # 1. 获取数值（处理 {'min': 35} 或直接是 35 的情况）
            val = v.get("min") if isinstance(v, dict) else v

            # 2. 尝试转换：只有能变成数字的才加入优化目标
            try:
                numeric_val = float(val)
                # 统一 key 的格式为 '7day', '28day', '56day'
                clean_key = k.replace("fc_", "").replace("day", "").strip() + "day"
                targets[clean_key] = numeric_val
            except (ValueError, TypeError):
                # 如果是 "C2" 这种字符串，float() 会报错，这里直接跳过
                # 这样它就不会干扰优化算法，但依然保留在 raw_req 里供 RAG 审计使用
                continue

        # 获取 Neo4j 规则
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
        with driver.session() as session:
            kg_rules = [r["rule"] for r in
                        session.run("MATCH (a)-[r]->(b) RETURN a.name+' '+type(r)+' '+b.name AS rule")]

        # 第二步：迭代闭环
        final_best = None
        for i in range(max_iters):
            print(f"\n[Iteration {i + 1}] Starting optimization search (add strain: {len(self.dynamic_constraints)} )")
            current_objs = raw_req.get("objectives", ["min_GWP"])
            if not current_objs:
                current_objs = ["min_GWP"]

            problem = IterativeMOOProblem(
                brain=self.brain,
                targets=targets,
                dynamic_constraints=self.dynamic_constraints,
                objectives_list=current_objs  # 传入动态目标
            )
            res_moo = minimize(problem, NSGA2(pop_size=200), ('n_gen', 100), seed=42)

            # --- 覆盖开始：用这段更稳健的逻辑替换你刚才那段 ---
            pop = res_moo.algorithm.pop
            X_all = pop.get("X")
            F_all = pop.get("F")
            G_all = pop.get("G")

            # 重新判断可行性 (基于 pop 里的原始数据)
            if G_all is None or G_all.size == 0:
                feasible = np.ones(len(X_all), dtype=bool)
            elif G_all.ndim == 1:
                feasible = (G_all <= 0)
            else:
                feasible = (G_all <= 0).all(axis=1)

            if not np.any(feasible):
                print("❌ Constraint conflict: No feasible solution found.")
                break

            # 排序获取索引
            f_feasible = F_all[feasible]
            if f_feasible.ndim == 1:
                candidate_indices = np.argsort(f_feasible)[:5]
            else:
                candidate_indices = np.argsort(f_feasible[:, 0])[:5]

            candidates = []
            for idx_local, idx_real in enumerate(candidate_indices):
                # 1. 这里的 actual_idx 必须对应 X_all 的行索引
                # np.where(feasible)[0] 拿到了所有可行解在原始种群中的位置
                all_feasible_indices = np.where(feasible)[0]
                actual_idx = all_feasible_indices[idx_real]

                # 2. 从我们定义的 X_all（种群全集）中提取
                current_x = X_all[actual_idx]

                # 3. 基础配比还原
                recipe = {
                    v: float(current_x[j] * (self.brain.maxs[v] - self.brain.mins[v]) + self.brain.mins[v])
                    for j, v in enumerate(self.brain.independent_vars)
                }

                # 4. 预测性能
                perf = self.brain.predict_full_workflow([recipe])

                # 5. 获取 GWP (关键修复：使用 F_all 而不是 res_moo.F)
                if F_all.ndim == 1:
                    # 单目标，F_all 是 [200,] 数组
                    current_gwp = float(F_all[actual_idx])
                else:
                    # 多目标，F_all 是 [200, n_obj] 矩阵
                    current_gwp = float(F_all[actual_idx, 0])

                recipe.update({
                    'GWP': current_gwp,
                    'w/b': float(perf['df_all']['w/b'][0]),
                    'SCM%': float(perf['df_all']['SCM%'][0]),
                    'p7': float(perf['p7'][0]),
                    'p28': float(perf['p28'][0]),
                    'p56': float(perf['p56'][0]),
                    'id': idx_local
                })
                candidates.append(recipe)

            self._print_candidate_table(candidates, i + 1)

            # LLM 专家审计
            audit = self.audit_candidates(candidates, kg_rules, self.brain.data_bounds)

            # --- 新增：查看 Normalizer 的真实输出 ---
            print(f"DEBUG - Normalizer status: {audit.get('status')}")
            print(f"DEBUG - Normalizer new constrains: {len(audit.get('new_constraints', []))}")
            # ------------------------------------

            # print(f"--- [DEBUG] 本轮 RAG 检索到的参考片段数量: {len(audit.get('debug_context', []))} ---")
            # solve 函数中改为：


            avg_wb_iter = np.mean([c['w/b'] for c in candidates])
            print(f"--- [DEBUG] mean w/b: {avg_wb_iter:.3f} ---")

            best_idx = audit.get('best_id', 0)

            try:
                best_idx = int(best_idx)
                if best_idx >= len(candidates) or best_idx < 0:
                    best_idx = 0
            except (ValueError, TypeError):
                best_idx = 0

                # 3. 更新当前最优方案
            final_best = candidates[best_idx]

            # 4. 统一判断状态 (支持 APPROVED, ACCEPTED 等)
            status_str = str(audit.get('status', "")).upper()
            if status_str in ['APPROVED', 'SUCCESS', 'ACCEPTED', 'ACCEPTABLE']:
                print(f"✅ LLM approves design (ID: {best_idx})")
                print(f"核心理由: {audit.get('reasoning', 'fits all design requirements')}")
                break
            # --- 核心修复结束 ---

            else:
                print(f"\n⚠️ LLM declines design。audit reasoning: {audit.get('reasoning', 'N/A')}")

                if audit.get('new_constraints'):
                    print("🆕 new constrains this iteration:")
                    added_count = 0
                    for nc in audit['new_constraints']:
                        # 检查是否是新规则
                        is_new = True
                        for existing in self.dynamic_constraints:
                            if existing['feature'] == nc['feature'] and existing['op'] == nc['op']:
                                is_new = False
                                break

                        if is_new:
                            # --- 核心改进：打印规律描述 ---
                            desc = nc.get('description', '基于材料学经验的修正')
                            source = nc.get('source', '专家经验/训练集边界')  # 核心修改：获取来源

                            print(f"   👉 New Physical Rules/Constraints Discovered: {desc}")
                            print(f"      Constraint: {nc['feature']} {nc['op']} {nc['val']}")
                            print(f"      Theoretical Basis: {source}")  # 打印来源

                            self.dynamic_constraints.append(nc)
                            added_count += 1

                    print(f"➕ Successfully injected {added_count} new physical rules(Total: {len(self.dynamic_constraints)})")
        if not final_best:
            print("⚠️ Max iterations reached without LLM approval. Returning the best GWP candidate.")
            # 这里的 candidates 是最后一轮生成的 15 个方案
            final_best = candidates[0]
        driver.close()
        return final_best

# --- 5. 启动执行 ---
if __name__ == "__main__":
    # 1. 先初始化 Brain
    brain = CatBoostChainedBrain()

    # 2. 初始化 RAG 检索器 (建议增加持久化逻辑)
    print("initial RAG knowledge base...")
    retriever = initialize_rag_db(API_KEY)

    # 3. 实例化 Manager 时必须传入 retriever
    manager = ConcreteAIManager(brain, retriever)

    query = input("Input your design needs: ")

    # 也可以设置一个默认值，防止用户直接按回车
    if not query.strip():
        query = "I need high late-age strength (28d>50MPa, 56d>65MPa), prioritize low carbon."

    best_recipe = manager.solve(query)

    # --- 修改最后的打印部分 ---
    if best_recipe:
        print("\n" + "=" * 25 + " The final design " + "=" * 25)
        print(f"{'component':<15} | {'dosage (lb/yd³)':>12}")  # 注意你的 LLM 提示词用的是 lb
        print("-" * 45)
        for v in manager.brain.independent_vars:
            print(f"{v:<15} | {best_recipe[v]:>12.2f}")

        print("-" * 45)
        print(f"{'GWP':<15} | {best_recipe['GWP']:>12.2f}")
        print(f"{'w/b':<15} | {best_recipe['w/b']:>12.3f}")
        print(f"{'SCM':<15} | {best_recipe['SCM%']:>12.2%}")
        # 改为打印具体的龄期强度
        print(f"{'7d strength':<15} | {best_recipe['p7']:>12.2f} MPa")
        print(f"{'28d strength':<15} | {best_recipe['p28']:>12.2f} MPa")
        print(f"{'56d strength':<15} | {best_recipe['p56']:>12.2f} MPa")
        print("=" * 60)
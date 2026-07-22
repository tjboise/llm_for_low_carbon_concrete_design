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
        source = doc.metadata.get('source', 'Standard_Doc').split('/')[-1]  # 只取文件名
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content.strip()[:1000]  # 每段强制限制 1000 字符
        context_parts.append(f"SOURCE_{i + 1}: [{source} | Page {page}]\nCONTENT: {content}")
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
        "target_strengths": "extracted from query",  # solve 里的 targets
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
        """向量化接口：一次性处理整个种群矩阵 x (shape: [pop_size, 11])"""
        # 1. 批量逆归一化 (使用 NumPy 的广播机制，极快)
        # x: [200, 11], mins: [11,], maxs: [11,]
        mins_arr = np.array([self.mins[v] for v in self.independent_vars])
        maxs_arr = np.array([self.maxs[v] for v in self.independent_vars])

        # 得到物理值矩阵
        phys_x = x * (maxs_arr - mins_arr) + mins_arr

        # 2. 将物理值转为列表字典格式，适配你原有的 predict_full_workflow
        raw_mixes = [
            {v: float(phys_x[i, j]) for j, v in enumerate(self.independent_vars)}
            for i in range(len(x))
        ]

        # 3. 批量预测
        perf = self.predict_full_workflow(raw_mixes)
        df = perf['df_all']  # 这是一个包含 200 行数据的 DataFrame

        # 4. 把 DataFrame 里的每一列都转成 NumPy 向量返回
        # 这样无论是原料 (PC, WATER) 还是派生量 (w/b, SCM%) 全都在里面了
        res_vectors = {col: df[col].values for col in df.columns}

        # 4. 返回 NumPy 数组格式，方便 _evaluate 批量处理
        res_vectors.update({
            'p7': np.array(perf['p7']),
            'p28': np.array(perf['p28']),
            'p56': np.array(perf['p56']),
            'gwp': np.array(perf['gwp'])
        })
        return res_vectors

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

from pymoo.core.problem import Problem


class IterativeMOOProblem(Problem):  # 改回 Problem
    def __init__(self, brain, targets, dynamic_constraints, objectives_list):
        self.brain = brain
        self.targets = targets
        self.dyn_cons = dynamic_constraints
        self.objectives_list = objectives_list

        n_constr = len(targets) + len(dynamic_constraints)
        super().__init__(
            n_var=len(brain.independent_vars),
            n_obj=len(objectives_list),
            n_constr=n_constr,
            xl=0, xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # 一次性拿到所有 200 个人的全参数预测结果
        res = self.brain.predict_full_workflow_for_moo(x)

        # --- 目标处理 ---
        f_list = []
        for obj in self.objectives_list:
            # 这里的 obj 对应 res 字典里的 key
            key_map = {"min_GWP": "gwp", "max_fc_7day": "p7", "max_fc_28day": "p28", "max_fc_56day": "p56"}
            val_vector = res.get(key_map.get(obj, obj))
            # NSGA-II 默认最小化，所以 max 目标要加负号
            f_list.append(val_vector if "min" in obj else -val_vector)
        out["F"] = np.column_stack(f_list)

        # --- 约束处理 ---
        g_list = []
        # 1. 强度硬约束
        for day_key, min_val in self.targets.items():
            clean_key = f'p{day_key.replace("day", "")}'
            g_list.append(min_val - res[clean_key])

        # 2. LLM 动态约束 (现在支持任何参数！)
        for con in self.dyn_cons:
            feature = con['feature']
            limit_val = float(con['val'])

            # 直接从 res 字典取向量，不管是 'w/b', 'PC' 还是 'WATER'
            if feature in res:
                curr_vals = res[feature]
                if con['op'] in ["<=", "<"]:
                    g_list.append(curr_vals - limit_val)
                elif con['op'] in [">=", ">"]:
                    g_list.append(limit_val - curr_vals)
            else:
                print(f"⚠️ Warning: Feature {feature} not found in prediction results.")

        out["G"] = np.column_stack(g_list)


# --- 4. 迭代管理器：协调 LLM 审计与算法搜索 ---
class ConcreteAIManager:
    def __init__(self, brain, retriever):
        self.brain = brain
        self.retriever = retriever  # RAG
        self.dynamic_constraints = []
        self.last_user_query = ""

    def run_topsis_decision(self, candidates, weights, objectives_list):
        """
        动态 TOPSIS：支持任意数量的优化目标
        weights: LLM 返回的权重列表，长度必须与 objectives_list 一致
        objectives_list: 比如 ["min_GWP", "max_fc_56day", "max_fc_7day"]
        """
        import numpy as np

        # 1. 动态构建目标矩阵 F
        # 根据 objectives_list 提取数据，如果是 max 则取正值，min 则取正值
        # TOPSIS 统一逻辑：寻找距离“最小值组合”最近的点（所以 max 目标我们要取负数）
        f_cols = []
        for obj in objectives_list:
            if obj == "min_GWP":
                f_cols.append([c['GWP'] for c in candidates])
            elif obj == "max_fc_7day":
                f_cols.append([-c['p7'] for c in candidates])
            elif obj == "max_fc_28day":
                f_cols.append([-c['p28'] for c in candidates])
            elif obj == "max_fc_56day":
                f_cols.append([-c['p56'] for c in candidates])

        F = np.array(f_cols).T  # 形状为 [候选数, 目标数]

        # 2. 向量归一化
        norm_F = F / (np.sqrt((F ** 2).sum(axis=0)) + 1e-8)

        # 3. 加权
        weighted_F = norm_F * np.array(weights)

        # 4. 确定理想解 (各列最小值) 和 负理想解 (各列最大值)
        ideal_best = weighted_F.min(axis=0)
        ideal_worst = weighted_F.max(axis=0)

        # 5. 计算得分
        dist_best = np.sqrt(((weighted_F - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_F - ideal_worst) ** 2).sum(axis=1))

        scores = dist_worst / (dist_best + dist_worst + 1e-8)
        return int(np.argmax(scores))

    def audit_candidates(self, candidates, kg_rules, data_bounds, current_objs):
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
                
                [ALLOWED FEATURES FOR CONSTRAINTS]
                You MUST ONLY use these exact keys for "feature":
                - "p7", "p28", "p56" (Strength targets)
                - "w/b", "SCM%", "TOTAL_BINDER", "b/a", "FA%", "SC%" (Mix ratios)
                - "PC", "FA", "SC", "SF", "WATER", "FAGG", "CAGG", "AEA", "WR_HR", "WR", "ACC" (Materials)

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
        # print("\n🔍 [DEBUG] Step 3: Auditor Original Output (Per Candidate):")
        # 尝试美化打印，如果解析失败则打印原文本
        # try:
        #     print(json.dumps(per_cand_results, indent=2, ensure_ascii=False))
        # except:
        #     print(aud_resp.text)

        # --- Step 4: Normalizer ---
        normalizer_prompt = f"""
        Summarize these audit results and determine decision weights for multi-objective balancing.

        [INPUT DATA]
        Audit Results: {json.dumps(per_cand_results, ensure_ascii=False)}
        User Query/Priority: {self.last_user_query}
        Current Optimization Objectives: {json.dumps(current_objs)}
        
        [STRICT NAMING RULES]
        1. All 'feature' names in 'new_constraints' MUST be chosen from this list: 
           ['p7', 'p28', 'p56', 'w/b', 'SCM%', 'FA%', 'SC%', 'TOTAL_BINDER', 'PC', 'FA', 'SC', 'SF', 'WATER', 'FAGG', 'CAGG', 'AEA', 'WR_HR', 'WR', 'ACC']
        2. DO NOT use human-readable names like 'fc_28day' or 'Cement'.
        3. If you see 'fc_28day' in the audit results, you MUST translate it back to 'p28' in the final JSON.

        [INSTRUCTIONS]
        1. Return a single JSON object with: "status", "best_id", "reasoning", "new_constraints", and "decision_weights".
        2. "decision_weights": You MUST provide a weight for each objective in {json.dumps(current_objs)}. 
           - The weights must sum to 1.0.
           - Adjust weights based on engineering logic: e.g., for massive structures, favor late-age strength (p56) and low GWP; for rapid repair, favor early strength (p7).
        3. "status": If all candidates meet standards and no new constraints are needed, return "APPROVED". Otherwise, "REJECTED".
        4. "best_id": This ID will be used for the FINAL selection if status is APPROVED.
        5. "reasoning": MUST include specific quotes and SOURCE_X citations. Explain WHY you chose the specific weights.

        [REVISED EXAMPLES]

Example 1 (Durability focused for Massive Pour):
    - Objectives: ["min_GWP", "max_fc_56day"]
    - User Context: "Massive bridge foundation, long-term durability is key."
    - Output JSON:
    {{
        "status": "APPROVED",
        "best_id": 3,
        "decision_weights": [0.4, 0.6],
        "new_constraints": [],
        "reasoning": "Approved per SOURCE_2 (ACI 207.1R): 'For massive concrete, minimize cementitious materials to control thermal rise.' I assigned 0.6 weight to 'p56' (max_fc_56day) to ensure long-term durability while maintaining a 0.4 weight for 'GWP' to keep cement content low."
    }}

Example 2 (Strict Strength Requirement & High Early Strength):
    - Objectives: ["min_GWP", "max_fc_7day", "max_fc_28day"]
    - User Context: "Need to strip forms in 3 days, strength must be >40MPa at 28d."
    - Output JSON:
    {{
        "status": "REJECTED",
        "best_id": null,
        "decision_weights": [0.2, 0.5, 0.3],
        "new_constraints": [
            {{"feature": "p28", "op": ">=", "val": 40}}
        ],
        "reasoning": "Rejected current candidates because 'p28' values are below the 40MPa threshold required by project specs. Since rapid construction is requested, I assigned 0.5 weight to 'p7' (max_fc_7day) to prioritize formwork removal efficiency."
    }}

Example 3 (Correction of AI Extrapolation):
    - Input Audits: ID 0: REJECTED. "w/b 0.85 exceeds experimental maximum."
    - Output JSON:
    {{
        "status": "REJECTED",
        "best_id": null,
        "decision_weights": [0.5, 0.5],
        "new_constraints": [
            {{"feature": "w/b", "op": "<=", "val": 0.714}}
        ],
        "reasoning": "Rejected candidates because 'w/b' ratio exceeds the empirical data bounds (0.714). Adding a physical constraint to force the optimizer back into the reliable model prediction range."
    }}
        """
        norm_resp = client.models.generate_content(model="gemini-2.0-flash", contents=normalizer_prompt)
        audit = json.loads(re.search(r'\{.*\}', norm_resp.text, re.DOTALL).group(0))

        # --- 在 Step 4 (norm_resp) 之后添加 ---
        # print("\n🔍 [DEBUG] Step 4: Normalizer Combined Output (Summary):")
        # try:
        #     print(json.dumps(audit, indent=2, ensure_ascii=False))
        # except:
        #     print(norm_resp.text)

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
        # 用于绘图的存储容器
        iteration_fronts = []  # 存储每一轮的所有可行解坐标
        all_injected_constraints = []  # 记录约束的变化
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

            if F_all.ndim > 1:
                # 提取当前轮次的所有可行解点 (GWP, -F_all[1]即强度)
                current_front = np.column_stack([F_all[feasible, 0], -F_all[feasible, 1]])
                iteration_fronts.append(current_front)

            if not np.any(feasible):
                print("❌ Constraint conflict: No feasible solution found.")
                break

            # 排序获取索引
            res = self.brain.predict_full_workflow_for_moo(X_all)
            f_feasible = F_all[feasible]
            all_feasible_indices = np.where(feasible)[0]

            # 1. 动态获取当前权重 (如果是第一轮迭代，audit 还没生成，使用默认值)
            # 注意：这里我们通过 locals() 检查变量是否存在，防止报错
            if 'audit' in locals() and audit.get('decision_weights'):
                current_weights = audit.get('decision_weights')
            else:
                current_weights = [1.0 / len(current_objs)] * len(current_objs)

            # 2. 构建一个临时的全量候选池，用于跑 TOPSIS
            full_front_candidates = []
            for i in range(len(f_feasible)):
                full_front_candidates.append({
                    'GWP': f_feasible[i, 0],
                    'p7': res['p7'][all_feasible_indices[i]],
                    'p28': res['p28'][all_feasible_indices[i]],
                    'p56': res['p56'][all_feasible_indices[i]]
                })

            # 3. 在整个 Pareto 前沿上运行 TOPSIS 找到数学上的“黄金点”
            global_best_idx = self.run_topsis_decision(full_front_candidates, current_weights, current_objs)

            # 4. 挑选代表性解给 LLM 审计：
            # - 数学冠军 (TOPSIS Best)
            # - GWP 最低点 (Most Eco)
            # - p56 最高点 (Most Durable)
            # - 以及冠军点附近的几个邻居点 (增加多样性)
            neighbors = []
            for offset in [-2, -1, 1, 2]:
                idx = global_best_idx + offset
                if 0 <= idx < len(f_feasible):
                    neighbors.append(idx)

            representative_indices = [global_best_idx, np.argmin(f_feasible[:, 0]),
                                      np.argmax(res['p56'][all_feasible_indices])] + neighbors

            # 去重并保持顺序
            candidate_indices = []
            for idx in representative_indices:
                if idx not in candidate_indices:
                    candidate_indices.append(idx)

            candidate_indices = candidate_indices[:10]  # 审计前 10 个最具代表性的解

            # 5. 生成真正的 candidates 列表（带原材料配比的）
            candidates = []
            for idx_local, idx_in_f in enumerate(candidate_indices):
                actual_pop_idx = all_feasible_indices[idx_in_f]
                current_x = X_all[actual_pop_idx]

                # 还原配比
                recipe = {
                    v: float(current_x[j] * (self.brain.maxs[v] - self.brain.mins[v]) + self.brain.mins[v])
                    for j, v in enumerate(self.brain.independent_vars)
                }

                # 获取该点的预测性能
                perf = self.brain.predict_full_workflow([recipe])

                recipe.update({
                    'GWP': f_feasible[idx_in_f, 0],
                    'w/b': float(perf['df_all']['w/b'][0]),
                    'SCM%': float(perf['df_all']['SCM%'][0]),
                    'p7': float(perf['p7'][0]),
                    'p28': float(perf['p28'][0]),
                    'p56': float(perf['p56'][0]),
                    'id': idx_local  # 这里的 id 是给 LLM 看的
                })
                candidates.append(recipe)

            self._print_candidate_table(candidates, i + 1)

            # LLM 专家审计

            audit = self.audit_candidates(candidates, kg_rules, self.brain.data_bounds, current_objs)


            if audit.get('new_constraints'):
                # 定义简单的修正字典
                quick_fix = {
                    "fc_28day": "p28", "28day": "p28", "fc_28d": "p28",
                    "fc_7day": "p7", "7day": "p7",
                    "fc_56day": "p56", "56day": "p56",
                    "Water/Binder": "w/b", "w/c": "w/b"
                }
                for nc in audit['new_constraints']:
                    # 自动替换不规范的名称
                    nc["feature"] = quick_fix.get(nc["feature"], nc["feature"])

            # --- 新增：查看 Normalizer 的真实输出 ---
            # print(f"DEBUG - Normalizer status: {audit.get('status')}")
            # print(f"DEBUG - Normalizer new constrains: {len(audit.get('new_constraints', []))}")
            # ------------------------------------

            # print(f"--- [DEBUG] 本轮 RAG 检索到的参考片段数量: {len(audit.get('debug_context', []))} ---")
            # solve 函数中改为：

            avg_wb_iter = np.mean([c['w/b'] for c in candidates])
            # print(f"--- [DEBUG] mean w/b: {avg_wb_iter:.3f} ---")

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
            dynamic_weights = audit.get('decision_weights')

            if status_str in ['APPROVED', 'SUCCESS', 'ACCEPTED', 'ACCEPTABLE']:
                # 检查权重有效性，如果 LLM 没给或者长度不对，给个默认均权
                if not dynamic_weights or len(dynamic_weights) != len(current_objs):
                    dynamic_weights = [1.0 / len(current_objs)] * len(current_objs)

                # --- 调用动态 TOPSIS 进行多目标筛选 ---
                # 注意传入当前的 current_objs 以对齐维度
                best_idx_topsis = self.run_topsis_decision(candidates, dynamic_weights, current_objs)
                final_best = candidates[best_idx_topsis]

                print(f"✅ LLM approves design via {len(current_objs)}-Objective TOPSIS.")
                print(f"⚖️ Weights Assigned by LLM: {dict(zip(current_objs, dynamic_weights))}")
                print(f"🎯 TOPSIS Selected Best Balance ID: {best_idx_topsis}")
                print(f"key reason: {audit.get('reasoning', 'N/A')}")
                break
            # --- 核心修复结束 ---

            else:
                print(f"\n⚠️ LLM declines design。audit reasoning: {audit.get('reasoning', 'N/A')}")

                if audit.get('new_constraints'):
                    print("🆕 Updating physical rules/constraints:")
                    added_count = 0
                    for nc in audit['new_constraints']:
                        updated = False
                        # 尝试在现有约束中寻找并更新
                        for existing in self.dynamic_constraints:
                            if existing['feature'] == nc['feature'] and existing['op'] == nc['op']:
                                old_val = float(existing['val'])
                                new_val = float(nc['val'])

                                # 如果是上限约束 (<=)，取最小值（越小越严）
                                if nc['op'] == '<=':
                                    if new_val < old_val:
                                        existing['val'] = new_val
                                        print(
                                            f"   📉 Tightened constraint: {nc['feature']} {nc['op']} {old_val} -> {new_val}")
                                        added_count += 1

                                # 如果是下限约束 (>=)，取最大值（越大越严）
                                elif nc['op'] == '>=':
                                    if new_val > old_val:
                                        existing['val'] = new_val
                                        print(
                                            f"   📈 Tightened constraint: {nc['feature']} {nc['op']} {old_val} -> {new_val}")
                                        added_count += 1

                                updated = True
                                break

                        # 如果是一个全新的特征约束，直接添加
                        if not updated:
                            print(
                                f"   👉 New Physical Rule: {nc['feature']} {nc['op']} {nc['val']} ({nc.get('description', 'N/A')})")
                            self.dynamic_constraints.append(nc)
                            added_count += 1

                    print(
                        f"➕ Successfully synchronized {added_count} rule updates (Total: {len(self.dynamic_constraints)})")
        if not final_best:
            print("⚠️ Max iterations reached without LLM approval. Returning the best GWP candidate.")

            final_best = candidates[0]

        if iteration_fronts and len(iteration_fronts) >= 1:
            import matplotlib.pyplot as plt
            import os
            plt.figure(figsize=(11, 7))

            # --- 1. 绘制背景对比 (Baseline) ---
            if os.path.exists("baseline_pareto_data.npy"):
                base_f = np.load("baseline_pareto_data.npy")
                b_idx = np.argsort(base_f[:, 0])
                plt.plot(base_f[b_idx, 0], base_f[b_idx, 1], color='#B0B0B0',
                         label='Traditional NSGA-II Front', linewidth=2, linestyle=':', alpha=0.6)
                plt.fill_between(base_f[b_idx, 0], base_f[b_idx, 1], color='#E0E0E0', alpha=0.2)

                if os.path.exists("baseline_best_point.npy"):
                    base_best = np.load("baseline_best_point.npy")
                    plt.scatter(base_best[0], base_best[1], color='#1F77B4', s=150,
                                marker='P', edgecolors='black', label='Traditional Best', zorder=5)

            # --- 2. 绘制 LIMOO 进化 (使用绿色系) ---
            # 使用 Greens 颜色映射，从浅绿到深绿
            colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(iteration_fronts)))

            for idx, front in enumerate(iteration_fronts):
                sorted_idx = np.argsort(front[:, 0])
                is_last = (idx == len(iteration_fronts) - 1)

                # 最后一轮用最深的绿色并加粗
                lw = 3 if is_last else 1.2
                plt.plot(front[sorted_idx, 0], front[sorted_idx, 1],
                         color=colors[idx], label=f'LIMOO Iteration {idx + 1}',
                         linewidth=lw, alpha=0.9, marker='o', markersize=3)

            # --- 3. 标注最终选定点 ---
            if final_best:
                plt.scatter(final_best['GWP'], final_best['p56'],
                            color='red', edgecolors='black', s=250, marker='*',
                            label='LIMOO Final Selection (Verified)', zorder=10)

            # 图表配置
            plt.xlabel("Global Warming Potential (GWP) [lb CO2-e/m³]", fontsize=12, fontweight='bold')
            plt.ylabel("56-day Compressive Strength [MPa]", fontsize=12, fontweight='bold')
            plt.title("Evolution of Pareto Front: LIMOO (Physics-Grounded) vs. Baseline", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.show()

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
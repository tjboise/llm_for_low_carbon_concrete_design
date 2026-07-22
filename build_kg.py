import os
import re
import fitz  # PyMuPDF
from google import genai
from neo4j import GraphDatabase
from pydantic import BaseModel

# ==========================================================
# 1. 配置信息 (API & Database)
# ==========================================================
client = genai.Client(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
MODEL_ID = "gemini-2.0-flash"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "Leke123123#"


# ==========================================================
# 2. 结构化定义
# ==========================================================
class Triple(BaseModel):
    subject: str
    relation: str
    object: str
    reason: str


class KnowledgeExtraction(BaseModel):
    triples: list[Triple]


# ==========================================================
# 3. 核心构建类
# ==========================================================
class LiteratureKGBuilder:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
            self.driver.verify_connectivity()
            print("[System] 成功连接至 Neo4j。")
        except Exception as e:
            print(f"[Fatal Error] 无法连接数据库: {e}")
            exit(1)

    def close(self):
        self.driver.close()

    def get_text_chunks(self, file_path, chunk_size=6000):
        """提取并初步清理PDF文本"""
        try:
            doc = fitz.open(file_path)
            full_text = "".join([page.get_text() for page in doc])
            full_text = re.sub(r'\s+', ' ', full_text)
            return [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        except Exception as e:
            print(f"[Error] 读取PDF失败: {e}")
            return []

    def normalize_entity(self, name):
        """Python端强制实体映射：锁定 SCM 核心范围"""
        name = name.upper().strip()
        # SCMs 严格限制
        if any(x in name for x in ["FLY ASH", "FA", "PFA", "PULVERIZED"]): return "FLY ASH"
        if any(x in name for x in ["SLAG", "GGBFS", "GGBS", "BLAST FURNACE"]): return "SLAG"
        if any(x in name for x in ["SILICA FUME", "SF", "MICRO-SILICA", "MICROSILICA"]): return "SILICA FUME"
        # 基础材料
        if any(x in name for x in ["CEMENT", "OPC", "PORTLAND", "BINDER"]): return "CEMENT"
        if any(x in name for x in ["WATER", "H2O"]): return "WATER"
        if any(x in name for x in ["AGGREGATE", "SAND", "GRAVEL", "STONE"]): return "AGGREGATE"
        # 性能指标
        if any(x in name for x in ["STRENGTH", "CS", "F'C", "MECHANICAL"]): return "COMPRESSIVE STRENGTH"
        if any(x in name for x in ["GWP", "CO2", "CARBON", "EMISSION", "FOOTPRINT"]): return "GWP"
        if any(x in name for x in ["DURABILITY", "SERVICE LIFE"]): return "DURABILITY"
        if any(x in name for x in ["WORKABILITY", "SLUMP", "FLOWABILITY"]): return "WORKABILITY"
        if any(x in name for x in ["W/B", "W/C", "RATIO"]): return "WATER-BINDER RATIO"

        return name

    def extract_and_upload(self, text_chunk):
        """带强约束的AI提取逻辑"""
        CORE_LIST = ["CEMENT", "FLY ASH", "SLAG", "SILICA FUME", "WATER", "AGGREGATE",
                     "COMPRESSIVE STRENGTH", "GWP", "DURABILITY", "WORKABILITY", "WATER-BINDER RATIO"]
        REL_LIST = ["IMPROVES", "REDUCES", "CAUSES", "REPLACES", "CORRELATES_WITH"]

        prompt = f"""
        You are a Concrete Research Expert. Extract triples STRICTLY following these rules:

        [1. CORE ENTITY WHITELIST]
        You MUST focus ONLY on these entities. Map all synonyms to these exact terms:
        - {', '.join(CORE_LIST)}

        [2. ALLOWED RELATIONS]
        You MUST ONLY use: {', '.join(REL_LIST)}

        [3. STRICT FILTERING]
        - DO NOT extract any other materials (e.g., Nano-particles, Fibers, or Chemicals).
        - If a relationship doesn't directly involve at least one core material and one property, DISCARD it.

        Text: {text_chunk}
        """

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={'response_mime_type': 'application/json', 'response_schema': KnowledgeExtraction}
            )
            data = response.parsed

            with self.driver.session() as session:
                for t in data.triples:
                    # 1. 关系清洗
                    rel = t.relation.strip().upper().replace(" ", "_")
                    if rel not in REL_LIST: rel = "CORRELATES_WITH"

                    # 2. 实体规范化
                    s_clean = self.normalize_entity(t.subject)
                    o_clean = self.normalize_entity(t.object)

                    # 3. 核心过滤：仅保留核心实体
                    if s_clean not in CORE_LIST and o_clean not in CORE_LIST:
                        continue

                    # --- 修复：提前处理转义，避开 Python f-string 语法限制 ---
                    safe_s = s_clean.replace("'", "\\'")
                    safe_o = o_clean.replace("'", "\\'")
                    safe_reason = t.reason.replace("'", "\\'").replace("\n", " ")

                    cypher = f"""
                    MERGE (s:Entity {{name: '{safe_s}'}})
                    MERGE (o:Entity {{name: '{safe_o}'}})
                    MERGE (s)-[r:{rel}]->(o)
                    ON CREATE SET r.reason = '{safe_reason}'
                    ON MATCH SET r.reason = r.reason + ' | ' + '{safe_reason}'
                    """
                    session.run(cypher)
            return len(data.triples)
        except Exception as e:
            print(f"  [Warning] Chunk skip: {e}")
            return 0


# ==========================================================
# 4. 执行流程
# ==========================================================
def main():
    builder = LiteratureKGBuilder()

    # 重置确认
    reset = input("\n是否清空 Neo4j 数据库以构建纯净的核心 SCM 知识网？(y/n): ").lower()
    if reset == 'y':
        with builder.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[System] 数据库已重置。")

    pdf_folder = "./PDFs"
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"[Info] 已创建 {pdf_folder} 文件夹。")
        return

    files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    for filename in files:
        print(f"\n[Document] {filename}")
        file_path = os.path.join(pdf_folder, filename)
        chunks = builder.get_text_chunks(file_path)

        total = 0
        for i, chunk in enumerate(chunks):
            # 这里的 \r 可以在控制台实时更新进度
            print(f"  -> Processing Chunk {i + 1}/{len(chunks)}...", end="\r")
            total += builder.extract_and_upload(chunk)
        print(f"\n[Success] {filename}: 提取了 {total} 条核心关系。")

    # 结果统计
    with builder.driver.session() as session:
        nodes = session.run("MATCH (n:Entity) RETURN count(n) as c").single()['c']
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']

    print("\n" + "=" * 50)
    print(f"核心图谱构建报告:")
    print(f" - 最终实体节点数: {nodes}")
    print(f" - 最终逻辑关系数: {rels}")
    print("=" * 50)
    print("建议前往 Neo4j Browser 运行: MATCH (n:Entity) RETURN n")

    builder.close()


if __name__ == "__main__":
    main()
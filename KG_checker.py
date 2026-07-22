import json
from google import genai
from neo4j import GraphDatabase

# ==========================================================
# 1. 配置信息
# ==========================================================
client = genai.Client(api_key="AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo")
MODEL_ID = "gemini-2.0-flash"
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "Leke123123#"

# 性能指标黑名单：这些节点绝对不能指向别人
PROPERTY_NODES = [
    'GWP', 'COMPRESSIVE STRENGTH', 'WORKABILITY', 'DURABILITY',
    'FLEXURAL STRENGTH', 'POROSITY', 'CARBON FOOTPRINT', 'SHRINKAGE', 'COST'
]


class KGDesignOptimizer:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
            self.driver.verify_connectivity()
            print("[System] 成功连接至 Neo4j 治理环境。")
        except Exception as e:
            print(f"[Fatal Error] 数据库连接失败: {e}")
            exit(1)

    def run_full_governance(self):
        print("\n" + "=" * 50)
        print("[System] 启动 V13.8 终极图谱治理流程")
        print("=" * 50)

        with self.driver.session() as session:
            # --- STEP 1: 删除因果倒置 ---
            print("\n[Phase 1] 正在修正因果流向...")
            session.run(f"MATCH (p:Entity)-[r]->(o:Entity) WHERE p.name IN {PROPERTY_NODES} DELETE r")
            print(f"  ✅ 已拦截所有从 {PROPERTY_NODES} 出发的错误关系。")

            # --- STEP 2: 实体清理（剔除噪音节点） ---
            print("\n[Phase 2] 正在清理非专业噪音实体...")
            nodes = list(session.run("MATCH (n:Entity) RETURN n.name as name"))
            for node in nodes:
                n_name = node['name']
                # 修复逻辑：预先处理引号
                safe_n_name = n_name.replace("'", "\\'")

                prompt = f"Is '{n_name}' a core material, property, or process in concrete design? (No for Table, Result, Method, etc.) Respond JSON: {{'keep': true/false}}"
                try:
                    resp = client.models.generate_content(model=MODEL_ID, contents=prompt,
                                                          config={'response_mime_type': 'application/json'})
                    decision = json.loads(resp.text)
                    if not decision.get('keep', True):
                        # 使用预处理好的 safe_n_name
                        session.run(f"MATCH (n:Entity {{name: '{safe_n_name}'}}) DETACH DELETE n")
                        print(f"  🗑️ 已移除无关实体: {n_name}")
                except Exception as e:
                    continue

            # --- STEP 3: 强力仲裁与唯一化 ---
            print("\n[Phase 3] 正在执行强力仲裁与关系综合...")
            pair_query = """
            MATCH (s:Entity)-[r]->(o:Entity)
            WITH s, o, count(r) as c
            WHERE c > 1
            RETURN s.name as s_n, o.name as o_n
            """
            pairs = list(session.run(pair_query))

            for pair in pairs:
                s_n, o_n = pair['s_n'], pair['o_n']
                safe_s = s_n.replace("'", "\\'")
                safe_o = o_n.replace("'", "\\'")

                # 提取矛盾证据
                rels_query = f"MATCH (s:Entity {{name: '{safe_s}'}})-[r]->(o:Entity {{name: '{safe_o}'}}) RETURN type(r) as t, r.reason as re"
                rels = list(session.run(rels_query))
                evidence = "\n".join([f"[{r['t']}]: {r['re']}" for r in rels])

                arbitration_prompt = f"""
                Conflict detected: '{s_n}' -> '{o_n}'. 
                Evidences: {evidence}
                Consolidate into ONE relationship. 
                Options: [IMPROVES, REDUCES, CAUSES, CONTAINS, INFLUENCES].
                Assign a 'net_impact_score' (-1.0 to 1.0).
                Respond in JSON: {{ "type": "TYPE", "score": 0.5, "reason": "summary" }}
                """

                try:
                    res_raw = client.models.generate_content(model=MODEL_ID, contents=arbitration_prompt,
                                                             config={'response_mime_type': 'application/json'})
                    res = json.loads(res_raw.text)

                    # 预处理理由中的引号
                    safe_reason = res['reason'].replace("'", "\\'")

                    # 执行删除并创建唯一关系
                    session.run(f"""
                        MATCH (s:Entity {{name: '{safe_s}'}})-[r]->(o:Entity {{name: '{safe_o}'}})
                        DELETE r
                        WITH s, o
                        MERGE (s)-[r2:{res['type']}]->(o)
                        SET r2.reason = '{safe_reason}', 
                            r2.impact_score = {res['score']},
                            r2.status = 'ARBITRATED'
                    """)
                    print(f"  ⚖️ [Arbitrated] {s_n} -> {o_n} 统一为: {res['type']}")
                except Exception as e:
                    continue

        print("\n" + "=" * 50 + "\n治理完成！")


if __name__ == "__main__":
    optimizer = KGDesignOptimizer()
    try:
        optimizer.run_full_governance()
    finally:
        optimizer.driver.close()
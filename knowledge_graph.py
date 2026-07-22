from neo4j import GraphDatabase

# ==========================================================
# 1. 配置信息
# ==========================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "Leke123123#"


class ConcreteKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_db(self):
        """清空数据库所有节点和关系"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared successfully.")

    def build_graph_no_mortar(self):
        cypher_query = """
        // 1. Materials
        MERGE (pc:Material {name: 'PC'}) SET pc.full_name = 'Portland Cement'
        MERGE (fa:Material {name: 'FA'}) SET fa.full_name = 'Fly Ash'
        MERGE (sc:Material {name: 'SC'}) SET sc.full_name = 'Slag Cement'
        MERGE (sf:Material {name: 'SF'}) SET sf.full_name = 'Silica Fume'
        MERGE (fagg:Material {name: 'FAGG'}) SET fagg.full_name = 'Fine Aggregate'
        MERGE (cagg:Material {name: 'CAGG'}) SET cagg.full_name = 'Coarse Aggregate'
        MERGE (water:Material {name: 'WATER'}) SET water.full_name = 'Water'

        // 2. Admixtures
        MERGE (wr:Admixture {name: 'WR'}) SET wr.full_name = 'Water Reducer'
        MERGE (wr_hr:Admixture {name: 'WR_HR'}) SET wr_hr.full_name = 'High-Range Water Reducer'
        MERGE (acc:Admixture {name: 'ACC'}) SET acc.full_name = 'Accelerator'
        MERGE (aea:Admixture {name: 'AEA'}) SET aea.full_name = 'Air Entrainer'

        // 3. Intermediate & Ratios (Mortar removed)
        MERGE (binder:Intermediate {name: 'Binder'})
        MERGE (paste:Intermediate {name: 'Paste'})
        MERGE (concrete:Intermediate {name: 'Concrete Matrix'})
        MERGE (wb:Ratio {name: 'w/b'})
        MERGE (scm_p:Ratio {name: 'SCM%'})
        MERGE (ba:Ratio {name: 'b/a'})

        // 4. Performance & Environment
        MERGE (gwp:Environmental {name: 'GWP'})
        MERGE (fc7:Performance {name: 'fc-7 day'})
        MERGE (fc28:Performance {name: 'fc-28day'})
        MERGE (fc56:Performance {name: 'fc-56day'})

        // 5. Relationships
        // Binder & SCM Logic
        MERGE (pc)-[:CONSTITUTES]->(binder)
        MERGE (fa)-[:CONSTITUTES]->(binder)
        MERGE (sc)-[:CONSTITUTES]->(binder)
        MERGE (sf)-[:CONSTITUTES]->(binder)
        MERGE (fa)-[:ASSOCIATED_WITH]->(scm_p)
        MERGE (sc)-[:ASSOCIATED_WITH]->(scm_p)
        MERGE (sf)-[:ASSOCIATED_WITH]->(scm_p)

        // Mixing logic (Paste -> Concrete Matrix)
        MERGE (binder)-[:FORMS]->(paste)
        MERGE (water)-[:FORMS]->(paste)
        MERGE (binder)-[:DEFINES]->(wb)
        MERGE (water)-[:DEFINES]->(wb)

        MERGE (paste)-[:FORMS]->(concrete)
        MERGE (fagg)-[:ADDED_TO]->(concrete)
        MERGE (cagg)-[:COMBINED_WITH]->(concrete)

        MERGE (binder)-[:REPRESENTS]->(ba)
        MERGE (fagg)-[:REPRESENTS]->(ba)
        MERGE (cagg)-[:REPRESENTS]->(ba)

        // Admixtures
        MERGE (wr)-[:OPTIMIZES]->(paste)
        MERGE (wr_hr)-[:OPTIMIZES]->(paste)
        MERGE (acc)-[:MIXED_WITH]->(binder)
        MERGE (aea)-[:MODIFIES]->(paste)

        // Performance & Environment
        MERGE (paste)-[:DETERMINES]->(fc28)
        MERGE (acc)-[:BOOSTS]->(fc7)
        MERGE (sf)-[:STRENGTHENS]->(fc28)
        MERGE (sc)-[:CONTRIBUTES_TO]->(fc28)
        MERGE (fa)-[:ENHANCES]->(fc56)
        MERGE (wb)-[:INVERSE_RELATION]->(fc28)

        MERGE (pc)-[:GENERATE]->(gwp)
        MERGE (scm_p)-[:REDUCES]->(gwp)
        """
        with self.driver.session() as session:
            session.run(cypher_query)
            print("Knowledge Graph has been rebuilt (Mortar node removed).")


if __name__ == "__main__":
    kg = ConcreteKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PWD)
    try:
        kg.clear_db()
        kg.build_graph_no_mortar()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        lb.close()
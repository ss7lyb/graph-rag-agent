import os
import json
from evaluator.retrieval_evaluator import GraphRAGRetrievalEvaluator
from agent.naive_rag_agent import NaiveRagAgent
from agent.hybrid_agent import HybridAgent
from agent.graph_agent import GraphAgent
from agent.deep_research_agent import DeepResearchAgent
from config.neo4jdb import get_db_manager

def main():
    print("初始化各种代理...")
    naive_agent = NaiveRagAgent()
    hybrid_agent = HybridAgent()
    graph_agent = GraphAgent()
    deep_agent = DeepResearchAgent()
    
    print("连接Neo4j数据库...")
    neo4j = get_db_manager().get_driver()
    
    # 创建配置
    config = {
        "naive_agent": naive_agent,
        "hybrid_agent": hybrid_agent,
        "graph_agent": graph_agent,
        "deep_agent": deep_agent,
        "neo4j_client": neo4j,
        "save_dir": "./evaluation_results",
        "metrics": [
            "retrieval_latency", 
            "retrieval_utilization", 
            "entity_coverage",
            "relationship_utilization",
            "graph_coverage",
            "community_relevance"
        ]
    }

    print("创建GraphRAG评估器...")
    evaluator = GraphRAGRetrievalEvaluator(config)
    
    # 测试问题
    questions = [
        "优秀学生申请的条件?",
        "德育分的评价标准?",
        "国家奖学金和国家励志奖学金互斥吗?",
    ]
    
    print("开始评估各代理性能...")
    results = evaluator.compare_agents(questions)
    
    print("保存评估结果...")
    results_path = os.path.join(config["save_dir"], "agent_comparison.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    comparison_table = evaluator.format_comparison_table(results)
    print("\n代理比较结果表格:")
    print(comparison_table)

    print("关闭资源...")
    naive_agent.close()
    hybrid_agent.close()
    graph_agent.close()
    deep_agent.close()

if __name__ == "__main__":
    main()
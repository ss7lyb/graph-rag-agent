from evaluator.composite_evaluator import CompositeGraphRAGEvaluator
from agent.naive_rag_agent import NaiveRagAgent
from agent.hybrid_agent import HybridAgent
from agent.graph_agent import GraphAgent
from agent.deep_research_agent import DeepResearchAgent
from config.neo4jdb import get_db_manager
from model.get_models import get_llm_model

def main():
    print("初始化各种代理...")
    naive_agent = NaiveRagAgent()
    hybrid_agent = HybridAgent()
    graph_agent = GraphAgent()
    deep_agent = DeepResearchAgent()
    
    print("连接Neo4j数据库...")
    neo4j = get_db_manager().get_driver()
    
    llm = get_llm_model()
    
    config = {
        "naive_agent": naive_agent,
        "hybrid_agent": hybrid_agent,
        "graph_agent": graph_agent,
        "deep_agent": deep_agent,
        "neo4j_client": neo4j,
        "llm": llm,
        "save_dir": "./evaluation_results_no_golden",
        "metrics": [
            "retrieval_latency", 
            "retrieval_utilization", 
            "entity_coverage",
            "relationship_utilization",
            "graph_coverage",
            "community_relevance",
            "response_coherence",
            "factual_consistency",
            "answer_comprehensiveness"
        ]
    }
    
    print("创建组合评估器...")
    evaluator = CompositeGraphRAGEvaluator(config)
    
    questions = [
        "优秀学生申请的条件?",
        "德育分的评价标准?",
        "国家奖学金和国家励志奖学金互斥吗?",
    ]
    
    print("开始评估各代理检索性能...")
    retrieval_results = evaluator.compare_retrieval_only(questions)
    
    retrieval_table = evaluator.format_comparison_table(retrieval_results)
    print("\n代理检索性能比较结果表格:")
    print(retrieval_table)

    print("保存各代理回答...")
    evaluator.save_agent_answers(questions)
    
    print("关闭资源...")
    naive_agent.close()
    hybrid_agent.close()
    graph_agent.close()
    deep_agent.close()

if __name__ == "__main__":
    main()
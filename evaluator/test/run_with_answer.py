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
        "save_dir": "./evaluation_results_with_golden",
        "metrics": [
            "em",  # 精确匹配
            "f1",  # F1分数
            "retrieval_latency", 
            "retrieval_utilization", 
            "entity_coverage",
            "relationship_utilization",
            "graph_coverage",
            "community_relevance"
        ]
    }
    
    print("创建组合评估器...")
    evaluator = CompositeGraphRAGEvaluator(config)
    
    questions = [
        "什么是优秀学生申请的条件?",
        "优秀学生干部的申请流程是什么?",
        "如何评选校级优秀学生?",
        "获得优秀学生称号的奖励标准是什么?"
    ]
    
    # 手动编写的标准答案
    golden_answers = [
        """
        优秀学生申请条件包括：
        1. 申请资格：必须是在校二、三、四年级本科生，热爱祖国，遵守法律法规。
        2. 学术要求：获得优秀奖学金二等奖及以上或民族班奖学金二等奖及以上，同时获社会工作奖B等及以上。
        3. 德育成绩：德育考核成绩需达到85分及以上。
        4. 综合素质：综合素质评价成绩需达到85分及以上。
        5. 思想品德：具备强烈的社会责任感，思想积极上进，关心集体，热心公益，尊敬师长，团结同学，文明礼貌，诚实守信。
        """,
        
        """
        优秀学生干部的申请流程包括以下步骤：
        1. 学校发布评选通知：根据相关文件启动评选活动。
        2. 个人申请阶段：学生通过学工系统在线申请，下载并填写评审表模板，上传表格附件后在线提交。
        3. 初步审核：辅导员审核在线填写意见并提交学院学生工作负责人。
        4. 学院评审：学院进行初评并将结果上报学校。
        5. 学校评审：学校对上报名单进行评审，确定获奖名单。
        6. 公示环节：评审结果在学院范围内公示，最终结果在全校范围内公示。
        申请者需注意同一学年只能选择一项申请，且提名后的名额不会递补。
        """,
        
        """
        校级优秀学生评选流程和标准：
        1. 评选对象：本科二、三、四年级学生，热爱祖国，遵守法律法规。
        2. 学术标准：学生在申请学年必须获得优秀奖学金二等奖及以上或民族班奖学金二等奖及以上。
        3. 综合考核：德育考核成绩需达到85分以上，综合素质评价成绩85分及以上。
        4. 评选程序：学生递交申请，经过学生民主评议，由相关组织部门审核批准。
        5. 上报流程：按规定比例推荐并上报先进事迹，确保评选过程公正透明。
        6. 诚信要求：如有弄虚作假或舞弊行为，一经核实，将取消获奖资格并追回所发放的奖学金。
        """,
        
        """
        优秀学生称号的奖励标准按级别区分：
        1. 全国"优秀学生"奖励1000元。
        2. 市"优秀学生标兵"奖励500元。
        3. 市"优秀学生"奖励500元。
        4. 校"优秀学生"奖励150元。
        除经济奖励外，获得优秀学生称号会记入学生的学籍档案，对未来就业和深造有积极影响。
        """
    ]
    
    print("开始评估各代理（使用标准答案）...")
    results = evaluator.compare_agents_with_golden_answers(questions, golden_answers)
    
    comparison_table = evaluator.format_comparison_table(results)
    print("\n代理比较结果表格（使用标准答案）:")
    print(comparison_table)

    print("关闭资源...")
    naive_agent.close()
    hybrid_agent.close()
    graph_agent.close()
    deep_agent.close()

if __name__ == "__main__":
    main()
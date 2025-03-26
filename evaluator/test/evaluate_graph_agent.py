import os
import json
import argparse
from evaluator.composite_evaluator import CompositeGraphRAGEvaluator
from agent.graph_agent import GraphAgent
from config.neo4jdb import get_db_manager
from model.get_models import get_llm_model

"""
python evaluate_graph_agent.py --questions_file=questions.json [--golden_answers_file=answer.json] [--verbose]
"""

def parse_args():
    parser = argparse.ArgumentParser(description="评估Graph Agent性能")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results/graph_agent",
                        help="评估结果保存目录")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="要评估的问题文件（JSON格式）")
    parser.add_argument("--golden_answers_file", type=str, default=None,
                        help="标准答案文件（JSON格式，可选）")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细评估过程")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("初始化Graph Agent和评估组件...")
    agent = GraphAgent()
    neo4j = get_db_manager().get_driver()
    llm = get_llm_model()
    
    # 配置评估器
    config = {
        "graph_agent": agent,
        "neo4j_client": neo4j,
        "llm": llm,
        "save_dir": args.save_dir,
        "debug": True  # 启用详细日志
    }
    
    # 创建评估器
    evaluator = CompositeGraphRAGEvaluator(config)
    
    # 加载问题
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # 提取问题列表
    if isinstance(questions_data, list):
        if isinstance(questions_data[0], dict) and "question" in questions_data[0]:
            questions = [item["question"] for item in questions_data]
        else:
            questions = questions_data
    else:
        questions = [questions_data]
    
    print(f"加载了{len(questions)}个问题待评估")
    
    # 如果有标准答案，执行带标准答案的评估
    if args.golden_answers_file:
        print("使用标准答案进行评估...")
        with open(args.golden_answers_file, "r", encoding="utf-8") as f:
            golden_answers = json.load(f)
        
        if isinstance(golden_answers, list):
            if isinstance(golden_answers[0], dict) and "answer" in golden_answers[0]:
                golden_answers = [item["answer"] for item in golden_answers]
        else:
            golden_answers = [golden_answers]
        
        if len(questions) != len(golden_answers):
            print(f"警告：问题数量({len(questions)})与标准答案数量({len(golden_answers)})不匹配")
            min_len = min(len(questions), len(golden_answers))
            questions = questions[:min_len]
            golden_answers = golden_answers[:min_len]
        
        results = evaluator.evaluate_with_golden_answers("graph", questions, golden_answers)
    else:
        print("仅评估检索性能...")
        results = evaluator.evaluate_retrieval_only("graph", questions)
    
    print("保存代理回答以供人工检查...")
    evaluator.save_agent_answers(questions, output_dir=os.path.join(args.save_dir, "answers"))
    
    print("\n评估结果:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    
    # 保存评估结果
    results_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"评估结果已保存至: {results_path}")
    
    agent.close()

if __name__ == "__main__":
    main()
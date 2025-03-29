import os
import sys
import json
import argparse
import time

# 添加父目录到路径，使得可以导入evaluator模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluator import set_debug_mode
from evaluator.utils.logging_utils import setup_logger
from evaluator.metrics import list_available_metrics
from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
from evaluator.evaluators.composite_evaluator import CompositeGraphRAGEvaluator

# 定义答案评估指标和检索评估指标
ANSWER_METRICS = ['em', 'f1', 'response_coherence', 'factual_consistency', 
                 'answer_comprehensiveness', 'llm_evaluation']

RETRIEVAL_METRICS = ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                    'entity_coverage', 'graph_coverage', 'relationship_utilization',
                    'community_relevance', 'subgraph_quality', 'chunk_utilization']

def get_available_metrics(metric_types):
    """
    获取可用的指标列表
    
    Args:
        metric_types: 指标类型列表，可以是 'answer', 'retrieval' 或 'all'
    
    Returns:
        List[str]: 可用指标列表
    """
    all_available = list_available_metrics()
    
    if 'all' in metric_types:
        return all_available
    
    available_metrics = []
    if 'answer' in metric_types:
        available_metrics.extend([m for m in ANSWER_METRICS if m in all_available])
    if 'retrieval' in metric_types:
        available_metrics.extend([m for m in RETRIEVAL_METRICS if m in all_available])
    
    return available_metrics

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
    parser.add_argument("--metrics", type=str, default="",
                        help="要评估的指标，用逗号分隔，留空则根据评估类型自动选择")
    parser.add_argument("--all", action="store_true",
                        help="是否使用所有可用指标进行评估")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置日志记录
    logger = setup_logger("evaluation", os.path.join(args.save_dir, "evaluation.log"))
    logger.info("开始评估Graph Agent")
    
    # 设置全局调试模式
    set_debug_mode(args.verbose)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 导入代理模块
    try:
        from agent.graph_agent import GraphAgent
        agent = GraphAgent()
        logger.info("成功加载Graph Agent")
    except ImportError:
        logger.error("无法导入GraphAgent，请确保agent模块存在")
        return
    
    # 导入Neo4j和LLM模块
    try:
        from config.neo4jdb import get_db_manager
        neo4j = get_db_manager().get_driver()
        logger.info("成功连接Neo4j数据库")
    except ImportError:
        logger.warning("无法连接Neo4j数据库，部分指标可能无法计算")
        neo4j = None
    
    try:
        from model.get_models import get_llm_model
        llm = get_llm_model()
        logger.info("成功加载LLM模型")
    except ImportError:
        logger.warning("无法加载LLM模型，部分指标可能无法计算")
        llm = None
    
    # 确定评估模式和指标
    has_golden_answers = args.golden_answers_file is not None
    
    # 确定使用哪些指标
    if args.all:
        # 使用所有可用指标
        metric_types = ['all']
        logger.info("使用所有可用指标进行评估")
    elif args.metrics:
        # 使用显式指定的指标
        metrics = args.metrics.split(",")
        logger.info(f"使用指定指标进行评估: {args.metrics}")
    else:
        # 根据评估模式自动选择指标
        if has_golden_answers:
            metric_types = ['answer', 'retrieval']
            logger.info("自动选择答案评估和检索评估指标")
        else:
            metric_types = ['retrieval']
            logger.info("仅自动选择检索评估指标")
    
    # 获取最终使用的指标列表
    if args.all or not args.metrics:
        metrics = get_available_metrics(metric_types)
    
    logger.info(f"最终使用的指标: {', '.join(metrics)}")
    
    # 配置评估器
    config = {
        "graph_agent": agent,
        "neo4j_client": neo4j,
        "llm": llm,
        "save_dir": args.save_dir,
        "debug": args.verbose,
        "metrics": metrics
    }
    
    evaluator_config = EvaluatorConfig(config)
    
    # 创建评估器
    evaluator = CompositeGraphRAGEvaluator(evaluator_config)
    logger.info(f"已创建评估器，使用指标数量: {len(metrics)}")
    
    # 然后再加载问题
    try:
        questions = evaluator.load_questions_from_file(args.questions_file)
        logger.info(f"已加载{len(questions)}个问题")
    except Exception as e:
        logger.error(f"加载问题文件出错: {e}")
        return
    
    start_time = time.time()
    
    # 如果有标准答案，执行带标准答案的评估
    if has_golden_answers:
        logger.info("使用标准答案进行评估...")
        try:
            golden_answers = evaluator.load_answers_from_file(args.golden_answers_file)
            
            if len(questions) != len(golden_answers):
                logger.warning(f"问题数量({len(questions)})与标准答案数量({len(golden_answers)})不匹配")
                min_len = min(len(questions), len(golden_answers))
                questions = questions[:min_len]
                golden_answers = golden_answers[:min_len]
                logger.info(f"截取了问题和答案，现在有{min_len}对问答")
            
            results = evaluator.evaluate_with_golden_answers("graph", questions, golden_answers)
        except Exception as e:
            logger.error(f"评估出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = {}
    else:
        logger.info("仅评估检索性能...")
        try:
            results = evaluator.evaluate_retrieval_only("graph", questions)
        except Exception as e:
            logger.error(f"评估出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = {}
    
    end_time = time.time()
    logger.info(f"评估完成，耗时: {end_time - start_time:.2f}秒")
    
    logger.info("保存代理回答以供人工检查...")
    evaluator.save_agent_answers(questions, output_dir=os.path.join(args.save_dir, "answers"))
    
    logger.info("\n评估结果:")
    for metric, score in results.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # 保存评估结果
    results_path = os.path.join(args.save_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存至: {results_path}")
    
    if hasattr(agent, 'close') and callable(agent.close):
        agent.close()
        logger.info("已关闭Agent")

if __name__ == "__main__":
    main()
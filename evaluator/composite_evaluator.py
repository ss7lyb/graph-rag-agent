from typing import Dict, List, Any
import os
import json

from evaluator.base import BaseEvaluator
from evaluator.answer_evaluator import AnswerEvaluator, AnswerEvaluationData, AnswerEvaluationSample
from evaluator.retrieval_evaluator import GraphRAGRetrievalEvaluator, RetrievalEvaluationData, RetrievalEvaluationSample
from evaluator.preprocessing import clean_references, clean_thinking_process

class CompositeGraphRAGEvaluator:
    """
    组合评估器，同时评估答案质量和检索性能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化组合评估器
        
        Args:
            config: 评估配置
        """
        self.config = config
        self.save_dir = config.get('save_dir', './evaluation_results')
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建答案评估器
        answer_config = config.copy()
        answer_config['save_dir'] = os.path.join(self.save_dir, 'answer_evaluation')
        answer_config['metrics'] = [m for m in config.get('metrics', []) if not m.startswith('retrieval_')]
        self.answer_evaluator = AnswerEvaluator(answer_config)
        
        # 创建检索评估器
        retrieval_config = config.copy()
        retrieval_config['save_dir'] = os.path.join(self.save_dir, 'retrieval_evaluation')
        retrieval_config['metrics'] = [m for m in config.get('metrics', []) if m.startswith('retrieval_') or m in ['entity_coverage', 'graph_coverage']]
        self.retrieval_evaluator = GraphRAGRetrievalEvaluator(retrieval_config)
        
        # 代理实例
        self.agents = {
            "naive": config.get("naive_agent"),
            "hybrid": config.get("hybrid_agent"),
            "graph": config.get("graph_agent"),
            "deep": config.get("deep_agent")
        }
        
    def evaluate_with_golden_answers(self, agent_name: str, questions: List[str], golden_answers: List[str]) -> Dict[str, float]:
        """
        使用标准答案评估特定代理
        
        Args:
            agent_name: 代理名称
            questions: 问题列表
            golden_answers: 标准答案列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        if len(questions) != len(golden_answers):
            raise ValueError("问题和标准答案数量不匹配")
            
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"未找到代理: {agent_name}")
        
        # 创建答案评估数据集
        answer_data = AnswerEvaluationData()
        
        # 创建检索评估数据集
        retrieval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for i, (question, golden_answer) in enumerate(zip(questions, golden_answers)):
            print(f"处理问题 {i+1}/{len(questions)}: {question[:30]}...")
            
            # 创建答案评估样本
            answer_sample = AnswerEvaluationSample(
                question=question,
                golden_answer=golden_answer
            )
            
            # 创建检索评估样本
            retrieval_sample = RetrievalEvaluationSample(
                question=question
            )

            # 记录开始时间
            import time
            start_time = time.time()
            
            # 普通回答
            answer = agent.ask(question)
            
            # 计算检索时间
            retrieval_time = time.time() - start_time
            
            # 更新样本
            answer_sample.update_system_answer(answer, agent_name)
            retrieval_sample.update_system_answer(answer, agent_name)
            retrieval_sample.retrieval_time = retrieval_time
            
            # 尝试使用Neo4j获取相关图数据（如果可用）
            neo4j_client = self.config.get('neo4j_client')
            if neo4j_client:
                entities, relationships = self.retrieval_evaluator._get_relevant_graph_data(question)
                retrieval_sample.update_retrieval_data(entities, relationships)
            
            # 添加到评估数据
            answer_data.append(answer_sample)
            retrieval_data.append(retrieval_sample)
        
        # 执行评估
        answer_results = self.answer_evaluator.evaluate(answer_data)
        retrieval_results = self.retrieval_evaluator.evaluate(retrieval_data)
        
        # 合并结果
        results = {**answer_results, **retrieval_results}
        
        # 保存合并结果
        results_path = os.path.join(self.save_dir, f"{agent_name}_evaluation.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def compare_agents_with_golden_answers(self, questions: List[str], golden_answers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        使用标准答案比较所有代理
        
        Args:
            questions: 问题列表
            golden_answers: 标准答案列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个代理的评估结果
        """
        if len(questions) != len(golden_answers):
            raise ValueError("问题和标准答案数量不匹配")
        
        results = {}
        
        for agent_name, agent in self.agents.items():
            if agent:
                print(f"评估代理: {agent_name}")
                agent_results = self.evaluate_with_golden_answers(agent_name, questions, golden_answers)
                results[agent_name] = agent_results
                
                # 打印结果
                print(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    print(f"  {metric}: {score:.4f}")
                print()
        
        # 保存比较结果
        results_path = os.path.join(self.save_dir, "agent_comparison_with_golden.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def evaluate_retrieval_only(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        仅评估检索性能
        
        Args:
            agent_name: 代理名称
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果
        """
        return self.retrieval_evaluator.evaluate_agent(agent_name, questions)
    
    def compare_retrieval_only(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        仅比较检索性能
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个代理的评估结果
        """
        return self.retrieval_evaluator.compare_agents(questions)
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为表格
        
        Args:
            results: 比较结果
            
        Returns:
            str: 表格字符串
        """
        # 获取所有指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 区分答案和检索指标
        answer_metrics = sorted([m for m in all_metrics if not m.startswith('retrieval_') and not m in ['entity_coverage', 'graph_coverage']])
        retrieval_metrics = sorted([m for m in all_metrics if m.startswith('retrieval_') or m in ['entity_coverage', 'graph_coverage']])
        
        # 构建表头
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        
        # 添加答案指标
        if answer_metrics:
            rows.append("| **答案质量指标** | " + " | ".join(["" for _ in results]) + " |")
            
            for metric in answer_metrics:
                row = f"| {metric} |"
                for agent in results:
                    score = results[agent].get(metric, "N/A")
                    if isinstance(score, float):
                        score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    row += f" {score_str} |"
                rows.append(row)
        
        # 添加检索指标
        if retrieval_metrics:
            rows.append("| **检索性能指标** | " + " | ".join(["" for _ in results]) + " |")
            
            for metric in retrieval_metrics:
                row = f"| {metric} |"
                for agent in results:
                    score = results[agent].get(metric, "N/A")
                    if isinstance(score, float):
                        score_str = f"{score:.4f}"
                    else:
                        score_str = str(score)
                    row += f" {score_str} |"
                rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table
    
    def save_agent_answers(self, questions: List[str], output_dir: str = None):
        """
        保存所有代理对问题的回答，便于人工评估
        
        Args:
            questions: 问题列表
            output_dir: 输出目录，默认为self.save_dir/answers
        """
        if output_dir is None:
            output_dir = os.path.join(self.save_dir, "answers")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for agent_name, agent in self.agents.items():
            if not agent:
                continue
                
            print(f"获取{agent_name}代理的回答...")
            answers = []
            
            for i, question in enumerate(questions):
                print(f"  问题 {i+1}/{len(questions)}")
                
                try:
                    # 获取回答
                    answer = agent.ask(question)
                    
                    # 清理回答
                    if agent_name == "deep":
                        answer = clean_thinking_process(answer)
                    answer = clean_references(answer)
                    
                    answers.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    print(f"  获取{agent_name}回答时出错: {e}")
                    answers.append({
                        "question": question,
                        "answer": f"获取回答时出错: {str(e)}"
                    })
            
            # 保存回答
            answers_path = os.path.join(output_dir, f"{agent_name}_answers.json")
            with open(answers_path, "w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=2)
            
            # 生成可读性更好的markdown格式
            markdown_path = os.path.join(output_dir, f"{agent_name}_answers.md")
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(f"# {agent_name.capitalize()} 代理的回答\n\n")
                
                for i, qa in enumerate(answers):
                    f.write(f"## 问题 {i+1}: {qa['question']}\n\n")
                    f.write(f"{qa['answer']}\n\n")
                    f.write("---\n\n")
            
            print(f"  已保存{agent_name}的回答到 {answers_path}")
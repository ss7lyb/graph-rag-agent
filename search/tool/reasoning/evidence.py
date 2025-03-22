from typing import Dict, List
import time
import hashlib

class EvidenceChainTracker:
    """
    证据链收集和推理跟踪器
    
    收集和管理深度研究过程中的证据链，
    追踪推理步骤使用的证据来源和推理逻辑
    """
    
    def __init__(self):
        """初始化证据链跟踪器"""
        self.reasoning_steps = []  # 推理步骤
        self.evidence_items = {}   # 证据项
        self.query_contexts = {}   # 查询上下文
        self.step_counter = 0      # 步骤计数器
        
    def start_new_query(self, query: str, keywords: Dict[str, List[str]]) -> str:
        """
        开始新的查询跟踪
        
        Args:
            query: 用户查询
            keywords: 查询关键词
            
        Returns:
            str: 查询ID
        """
        # 生成查询ID
        query_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()[:10]
        
        # 存储查询上下文
        self.query_contexts[query_id] = {
            "query": query,
            "keywords": keywords,
            "start_time": time.time(),
            "step_ids": []
        }
        
        return query_id
    
    def add_reasoning_step(self, 
                         query_id: str, 
                         search_query: str, 
                         reasoning: str) -> str:
        """
        添加推理步骤
        
        Args:
            query_id: 查询ID
            search_query: 搜索查询
            reasoning: 推理过程
            
        Returns:
            str: 步骤ID
        """
        # 生成步骤ID
        step_id = f"step_{self.step_counter}"
        self.step_counter += 1
        
        # 创建步骤记录
        step = {
            "step_id": step_id,
            "query_id": query_id,
            "search_query": search_query,
            "reasoning": reasoning,
            "evidence_ids": [],
            "timestamp": time.time()
        }
        
        # 添加步骤到列表并关联到查询
        self.reasoning_steps.append(step)
        if query_id in self.query_contexts:
            self.query_contexts[query_id]["step_ids"].append(step_id)
        
        return step_id
    
    def add_evidence(self, 
                   step_id: str, 
                   source_id: str, 
                   content: str, 
                   source_type: str) -> str:
        """
        添加证据项
        
        Args:
            step_id: 步骤ID
            source_id: 来源ID（如块ID）
            content: 证据内容
            source_type: 来源类型
            
        Returns:
            str: 证据ID
        """
        # 生成证据ID
        evidence_id = hashlib.md5(f"{source_id}:{content[:50]}".encode()).hexdigest()[:10]
        
        # 创建证据记录
        evidence = {
            "evidence_id": evidence_id,
            "source_id": source_id,
            "content": content,
            "source_type": source_type,
            "timestamp": time.time()
        }
        
        # 存储证据并关联到步骤
        self.evidence_items[evidence_id] = evidence
        
        # 查找步骤并添加证据ID
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                if evidence_id not in step["evidence_ids"]:
                    step["evidence_ids"].append(evidence_id)
                break
        
        return evidence_id
    
    def get_reasoning_chain(self, query_id: str) -> List[Dict]:
        """
        获取完整的推理链
        
        Args:
            query_id: 查询ID
            
        Returns:
            List[Dict]: 推理链，包含步骤和证据
        """
        if query_id not in self.query_contexts:
            return []
        
        # 获取查询相关的步骤ID
        step_ids = self.query_contexts[query_id]["step_ids"]
        
        # 按时间顺序收集步骤
        steps = []
        for step_id in step_ids:
            for step in self.reasoning_steps:
                if step["step_id"] == step_id:
                    # 复制步骤并添加完整证据
                    step_copy = step.copy()
                    step_copy["evidence"] = []
                    
                    # 添加证据详情
                    for evidence_id in step["evidence_ids"]:
                        if evidence_id in self.evidence_items:
                            step_copy["evidence"].append(
                                self.evidence_items[evidence_id]
                            )
                    
                    steps.append(step_copy)
                    break
        
        # 按时间戳排序
        steps.sort(key=lambda x: x["timestamp"])
        
        # 添加查询上下文
        result = {
            "query": self.query_contexts[query_id]["query"],
            "keywords": self.query_contexts[query_id]["keywords"],
            "start_time": self.query_contexts[query_id]["start_time"],
            "end_time": time.time(),
            "steps": steps
        }
        
        return result
    
    def get_step_evidence(self, step_id: str) -> List[Dict]:
        """
        获取特定步骤的证据
        
        Args:
            step_id: 步骤ID
            
        Returns:
            List[Dict]: 证据列表
        """
        # 查找步骤
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                # 收集证据
                evidence_list = []
                for evidence_id in step["evidence_ids"]:
                    if evidence_id in self.evidence_items:
                        evidence_list.append(
                            self.evidence_items[evidence_id]
                        )
                return evidence_list
        
        return []
    
    def summarize_reasoning(self, query_id: str) -> Dict:
        """
        总结推理过程
        
        Args:
            query_id: 查询ID
            
        Returns:
            Dict: 推理摘要
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"summary": "没有找到相关推理链"}
        
        # 计算统计信息
        steps_count = len(chain.get("steps", []))
        evidence_count = sum(len(step.get("evidence", [])) 
                           for step in chain.get("steps", []))
        
        # 识别关键步骤（有最多证据的步骤）
        key_steps = []
        if steps_count > 0:
            # 按证据数量排序
            sorted_steps = sorted(
                chain.get("steps", []),
                key=lambda x: len(x.get("evidence", [])),
                reverse=True
            )
            
            # 取前3个关键步骤
            key_steps = sorted_steps[:min(3, len(sorted_steps))]
        
        # 计算处理时间
        duration = chain.get("end_time", time.time()) - chain.get("start_time", time.time())
        
        # 生成摘要
        summary = {
            "query": chain.get("query", ""),
            "steps_count": steps_count,
            "evidence_count": evidence_count,
            "duration_seconds": duration,
            "key_steps": [
                {
                    "step_id": step.get("step_id"),
                    "search_query": step.get("search_query"),
                    "evidence_count": len(step.get("evidence", []))
                }
                for step in key_steps
            ]
        }
        
        return summary
    
    def get_evidence_source_stats(self, query_id: str) -> Dict:
        """
        获取证据来源统计
        
        Args:
            query_id: 查询ID
            
        Returns:
            Dict: 证据来源统计
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"sources": {}}
        
        # 收集所有证据
        all_evidence = []
        for step in chain.get("steps", []):
            all_evidence.extend(step.get("evidence", []))
        
        # 按来源类型分组
        sources = {}
        for evidence in all_evidence:
            source_type = evidence.get("source_type", "unknown")
            if source_type not in sources:
                sources[source_type] = 0
            sources[source_type] += 1
        
        return {"sources": sources}
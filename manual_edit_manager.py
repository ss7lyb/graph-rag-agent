import time
from datetime import datetime
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table
from config.neo4jdb import get_db_manager

class ManualEditManager:
    """
    手动编辑同步管理器，负责识别、保留和处理Neo4j数据库中的手动编辑。
    
    主要功能：
    1. 识别手动编辑的节点和关系
    2. 保留手动编辑，确保增量更新不会覆盖它
    3. 解决自动更新和手动编辑之间的冲突
    """
    
    def __init__(self):
        """初始化手动编辑同步管理器"""
        self.console = Console()
        self.graph = get_db_manager().graph
        
        # 性能计时器
        self.detection_time = 0
        self.sync_time = 0
        
        # 编辑统计
        self.edit_stats = {
            "manual_entities": 0,
            "manual_relations": 0,
            "preserved_edits": 0,
            "conflicts_resolved": 0
        }
    
    def _setup_manual_edit_tracking(self):
        """
        设置手动编辑追踪，确保数据库支持追踪手动编辑
        """
        try:
            # 添加时间戳触发器追踪节点和关系的创建/修改时间
            self.graph.query("""
            CALL apoc.trigger.add(
                'updateTimestamps',
                'UNWIND apoc.trigger.nodesByLabel({},"__Entity__") AS n 
                 SET n.updated_at = datetime(), 
                     n.created_at = CASE WHEN exists(n.created_at) THEN n.created_at ELSE datetime() END',
                {phase: 'after'}
            )
            """)
            
            self.console.print("[green]成功设置节点时间戳追踪[/green]")
            
            # 添加关系时间戳触发器
            self.graph.query("""
            CALL apoc.trigger.add(
                'updateRelTimestamps',
                'UNWIND apoc.trigger.relationships({}) AS r 
                 SET r.updated_at = datetime(), 
                     r.created_at = CASE WHEN exists(r.created_at) THEN r.created_at ELSE datetime() END',
                {phase: 'after'}
            )
            """)
            
            self.console.print("[green]成功设置关系时间戳追踪[/green]")
            
        except Exception as e:
            self.console.print(f"[yellow]设置手动编辑追踪时出错 (可能APOC未安装): {e}[/yellow]")
            self.console.print("[yellow]将使用基础的手动编辑检测方法[/yellow]")

    def detect_manual_edits(self) -> Dict[str, int]:
        """
        检测数据库中的手动编辑
        
        Returns:
            Dict: 手动编辑的统计
        """
        start_time = time.time()
        
        # 1. 检测手动创建的实体节点
        manual_entity_query = """
        MATCH (e:`__Entity__`)
        WHERE e.manual_edit = true 
           OR e.created_by IS NOT NULL
           OR e.edited_by IS NOT NULL
        RETURN count(e) AS manual_entities
        """
        
        entity_result = self.graph.query(manual_entity_query)
        manual_entities = entity_result[0]["manual_entities"] if entity_result else 0
        
        # 2. 检测手动创建的关系
        manual_relation_query = """
        MATCH ()-[r]->()
        WHERE r.manual_edit = true 
           OR r.created_by IS NOT NULL
           OR r.edited_by IS NOT NULL
        RETURN count(r) AS manual_relations
        """
        
        relation_result = self.graph.query(manual_relation_query)
        manual_relations = relation_result[0]["manual_relations"] if relation_result else 0
        
        # 3. 检测通过时间戳识别的可能手动编辑
        # 如果支持时间戳追踪，则可以通过比较创建时间和文档处理时间来识别
        timestamp_query = """
        MATCH (e:`__Entity__`) 
        WHERE exists(e.created_at) 
          AND NOT exists(e.generated_by_system)
        RETURN count(e) AS timestamp_entities
        """
        
        try:
            timestamp_result = self.graph.query(timestamp_query)
            timestamp_entities = timestamp_result[0]["timestamp_entities"] if timestamp_result else 0
        except:
            timestamp_entities = 0
        
        # 更新统计
        self.edit_stats["manual_entities"] = manual_entities
        self.edit_stats["manual_relations"] = manual_relations
        
        # 计算检测时间
        self.detection_time = time.time() - start_time
        
        # 显示检测结果
        self.console.print(f"[blue]手动编辑检测完成，耗时: {self.detection_time:.2f}秒[/blue]")
        self.console.print(f"[blue]检测到 {manual_entities} 个手动编辑的实体节点，"
                          f"{manual_relations} 个手动编辑的关系[/blue]")
        
        return {
            "manual_entities": manual_entities,
            "manual_relations": manual_relations,
            "timestamp_entities": timestamp_entities
        }
    
    def mark_manual_edit(self, entity_id: str, edit_info: Dict[str, Any]) -> bool:
        """
        标记实体为手动编辑
        
        Args:
            entity_id: 实体ID
            edit_info: 编辑信息
            
        Returns:
            bool: 标记是否成功
        """
        # 准备编辑信息
        params = {
            "entity_id": entity_id,
            "edited_by": edit_info.get("edited_by", "manual"),
            "edit_time": edit_info.get("edit_time", datetime.now().isoformat()),
            "edit_comment": edit_info.get("edit_comment", ""),
            "manual_edit": True
        }
        
        # 执行标记查询
        query = """
        MATCH (e:`__Entity__` {id: $entity_id})
        SET e.manual_edit = $manual_edit,
            e.edited_by = $edited_by,
            e.edit_time = $edit_time,
            e.edit_comment = $edit_comment
        RETURN e.id AS entity_id
        """
        
        try:
            result = self.graph.query(query, params=params)
            return bool(result and result[0]["entity_id"])
        except Exception as e:
            self.console.print(f"[red]标记实体为手动编辑时出错: {e}[/red]")
            return False
    
    def preserve_manual_edits(self, changed_files: List[str]) -> int:
        """
        确保增量更新不会覆盖手动编辑
        
        Args:
            changed_files: 变更的文件列表
            
        Returns:
            int: 保留的手动编辑数量
        """
        start_time = time.time()
        
        # 1. 标记与变更文件相关的手动编辑节点
        preserve_query = """
        MATCH (d:`__Document__`)<-[:PART_OF]-(c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
        WHERE d.fileName IN $changed_files
          AND (e.manual_edit = true OR e.created_by IS NOT NULL OR e.edited_by IS NOT NULL)
        SET e.preserve_edit = true
        RETURN count(e) AS preserved_count
        """
        
        try:
            result = self.graph.query(preserve_query, params={"changed_files": changed_files})
            preserved_count = result[0]["preserved_count"] if result else 0
        except Exception as e:
            self.console.print(f"[yellow]标记保留手动编辑时出错: {e}[/yellow]")
            preserved_count = 0
        
        # 2. 创建保护规则，防止删除手动编辑的节点
        protection_query = """
        MATCH (e:`__Entity__`) 
        WHERE e.preserve_edit = true
        SET e.protected = true
        RETURN count(e) AS protected_count
        """
        
        try:
            protection_result = self.graph.query(protection_query)
            protected_count = protection_result[0]["protected_count"] if protection_result else 0
        except Exception as e:
            self.console.print(f"[yellow]创建保护规则时出错: {e}[/yellow]")
            protected_count = 0
        
        # 更新统计信息
        self.edit_stats["preserved_edits"] = preserved_count
        
        # 计算同步时间
        self.sync_time = time.time() - start_time
        
        self.console.print(f"[blue]手动编辑保护完成，耗时: {self.sync_time:.2f}秒[/blue]")
        self.console.print(f"[blue]已保护 {preserved_count} 个手动编辑，"
                          f"{protected_count} 个节点被标记为受保护[/blue]")
        
        return preserved_count
    
    def resolve_conflicts(self, conflict_strategy: str = "manual_first") -> int:
        """
        解决自动更新和手动编辑之间的冲突
        
        Args:
            conflict_strategy: 冲突解决策略，可以是 "manual_first"（优先保留手动编辑），
                              "auto_first"（优先自动更新）或 "merge"（尝试合并）
            
        Returns:
            int: 解决的冲突数量
        """
        start_time = time.time()
        
        # 查找可能的冲突节点
        conflict_query = """
        MATCH (e:`__Entity__`)
        WHERE (e.manual_edit = true OR e.edited_by IS NOT NULL)
          AND e.system_generated = true
        RETURN e.id AS entity_id, e.type AS entity_type
        """
        
        try:
            conflicts = self.graph.query(conflict_query)
        except Exception as e:
            self.console.print(f"[yellow]查找冲突节点时出错: {e}[/yellow]")
            conflicts = []
        
        resolved_count = 0
        
        # 基于策略解决冲突
        if conflicts:
            for conflict in conflicts:
                entity_id = conflict["entity_id"]
                
                if conflict_strategy == "manual_first":
                    # 保留手动编辑，移除自动生成标记
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'manual_preferred',
                        e.system_generated = null
                    RETURN e.id
                    """
                
                elif conflict_strategy == "auto_first":
                    # 优先自动更新，移除手动编辑标记
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'auto_preferred',
                        e.manual_edit = null,
                        e.edited_by = null
                    RETURN e.id
                    """
                
                else:  # "merge" 策略
                    # 尝试合并两种编辑
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'merged',
                        e.merged_at = datetime()
                    RETURN e.id
                    """
                
                try:
                    result = self.graph.query(resolution_query, params={"entity_id": entity_id})
                    if result and result[0]:
                        resolved_count += 1
                except Exception as e:
                    self.console.print(f"[red]解决实体 {entity_id} 的冲突时出错: {e}[/red]")
        
        # 更新统计信息
        self.edit_stats["conflicts_resolved"] = resolved_count
        
        resolution_time = time.time() - start_time
        
        self.console.print(f"[blue]冲突解决完成，耗时: {resolution_time:.2f}秒[/blue]")
        self.console.print(f"[blue]已解决 {resolved_count} 个冲突，使用策略: {conflict_strategy}[/blue]")
        
        return resolved_count
    
    def display_edit_stats(self):
        """显示手动编辑统计"""
        stats_table = Table(title="手动编辑统计")
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("值", justify="right")
        
        for key, value in self.edit_stats.items():
            stats_table.add_row(key, str(value))
        
        self.console.print(stats_table)
        
        # 显示时间统计
        self.console.print(f"[blue]检测耗时: {self.detection_time:.2f}秒, "
                          f"同步耗时: {self.sync_time:.2f}秒[/blue]")
        
    def process(self, changed_files: List[str], conflict_strategy: str = "manual_first") -> Dict[str, Any]:
        """
        执行完整的手动编辑同步流程
        
        Args:
            changed_files: 变更的文件列表
            conflict_strategy: 冲突解决策略
            
        Returns:
            Dict: 处理结果统计
        """
        try:
            # 设置手动编辑追踪
            self._setup_manual_edit_tracking()
            
            # 检测手动编辑
            edit_stats = self.detect_manual_edits()
            
            # 保留手动编辑
            preserved_count = self.preserve_manual_edits(changed_files)
            
            # 解决冲突
            resolved_count = self.resolve_conflicts(conflict_strategy)
            
            # 显示统计
            self.display_edit_stats()
            
            return {
                "detection_time": self.detection_time,
                "sync_time": self.sync_time,
                "edit_stats": self.edit_stats,
                "preserved_count": preserved_count,
                "resolved_count": resolved_count
            }
            
        except Exception as e:
            self.console.print(f"[red]手动编辑同步过程中出现错误: {e}[/red]")
            raise
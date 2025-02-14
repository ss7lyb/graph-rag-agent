import os
from dotenv import load_dotenv
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from model.get_models import get_llm_model, get_embeddings_model
from config.prompt import (
    system_template_build_graph,
    human_template_build_graph,
    community_template
)
from config.settings import (
    entity_types,
    relationship_types,
    theme,
    FILES_DIR,
    CHUNK_SIZE,
    OVERLAP,
    community_algorithm
)
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from graph.struct_builder import GraphStructureBuilder
from graph.entity_extractor import EntityRelationExtractor
from graph.graph_writer import GraphWriter
from graph.entity_indexer import EntityIndexManager
from graph.similar_entity import GDSConfig, SimilarEntityDetector
from graph.entity_merger import EntityMerger
from community.dector import LeidenDetector, SLLPADetector
from community.summary import CommunitySummarizerFactory
from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph

import shutup
shutup.please()

class KnowledgeGraphProcessor:
    """
    知识图谱处理器，整合了图谱构建和索引处理的完整流程。
    
    主要功能包括：
    1. 知识图谱的基础构建（文件读取、分块、实体抽取等）
    2. 实体索引的创建和管理
    3. 相似实体的检测和合并
    4. 社区检测和摘要生成
    """
    
    def __init__(self):
        """初始化知识图谱处理器"""
        # 初始化终端界面
        self.console = Console()
        self.file_contents = []
        
        # 加载环境变量
        load_dotenv()
        
        # 初始化组件
        self._initialize_components()

    def _create_progress(self):
        """创建进度显示器"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _initialize_components(self):
        """初始化所有必要的组件"""
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=6)
            
            # 初始化模型
            self.llm = get_llm_model()
            self.embeddings = get_embeddings_model()
            progress.advance(task)
            
            # 初始化图数据库连接
            self.gds = GraphDataScience(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
            )
            self.graph = Neo4jGraph()
            progress.advance(task)
            
            # 初始化文本处理器
            self.file_reader = FileReader(FILES_DIR)
            self.chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)
            progress.advance(task)
            
            # 初始化图谱构建器
            self.struct_builder = GraphStructureBuilder()
            self.entity_extractor = EntityRelationExtractor(
                self.llm,
                system_template_build_graph,
                human_template_build_graph,
                entity_types,
                relationship_types
            )
            progress.advance(task)
            
            # 初始化索引管理器
            self.index_manager = EntityIndexManager()
            self.gds_config = GDSConfig()
            self.entity_detector = SimilarEntityDetector(self.gds_config)
            self.entity_merger = EntityMerger()
            progress.advance(task)
            
            # 初始化结果存储
            self.processing_results = {
                'graph_build': {},
                'index_process': {}
            }
            progress.advance(task)

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格"""
        table = Table(title=title, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")
        
        for key, value in data.items():
            table.add_row(key, str(value))
        
        self.console.print(table)

    def build_base_graph(self):
        """构建基础知识图谱"""
        self._display_stage_header("构建基础知识图谱")
        
        try:
            # 1. 读取文件
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]读取文件...", total=1)
                self.file_contents = self.file_reader.read_txt_files()
                progress.update(task, completed=1)
                
                # 显示文件信息
                table = Table(title="文件信息")
                table.add_column("文件名")
                table.add_column("内容长度", justify="right")
                for file_name, content in self.file_contents:
                    table.add_row(file_name, str(len(content)))
                self.console.print(table)
            
            # 2. 文本分块
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]文本分块...", total=len(self.file_contents))
                for file_content in self.file_contents:
                    chunks = self.chunker.chunk_text(file_content[1])
                    file_content.append(chunks)
                    progress.advance(task)
            
            # 3. 构建图结构
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]构建图结构...", total=3)
                
                # 清空并创建Document节点
                self.struct_builder.clear_database()
                for file_content in self.file_contents:
                    self.struct_builder.create_document(
                        type="local",
                        uri=str(FILES_DIR),
                        file_name=file_content[0],
                        domain=theme
                    )
                progress.advance(task)
                
                # 创建Chunk节点和关系
                for file_content in self.file_contents:
                    result = self.struct_builder.create_relation_between_chunks(
                        file_content[0],
                        file_content[2]
                    )
                    file_content.append(result)
                progress.advance(task)
                progress.advance(task)
            
            # 4. 提取实体和关系
            with self._create_progress() as progress:
                total_chunks = sum(len(file_content[2]) for file_content in self.file_contents)
                task = progress.add_task("[cyan]提取实体和关系...", total=total_chunks)
                
                def progress_callback(chunk_index):
                    progress.advance(task)
                
                self.file_contents = self.entity_extractor.process_chunks(
                    self.file_contents,
                    progress_callback
                )
            
            # 5. 写入数据库
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]写入数据库...", total=1)
                graph_writer = GraphWriter(self.graph)
                graph_writer.process_and_write_graph_documents(self.file_contents)
                progress.update(task, completed=1)
            
            self.console.print("[green]基础知识图谱构建完成[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]基础图谱构建失败: {str(e)}[/red]")
            return False

    def build_index_and_communities(self):
        """构建索引和社区"""
        self._display_stage_header("构建索引和社区")
        
        try:
            # 1. 创建实体索引
            if not self.index_manager.create_entity_index():
                raise Exception("实体索引创建失败")
            
            # 2. 检测和合并相似实体
            duplicates = self.entity_detector.process_entities()
            merged_count = self.entity_merger.process_duplicates(duplicates)
            self._display_results_table(
                "实体合并结果",
                {"合并的实体组数": merged_count}
            )
            
            # 3. 社区检测
            detector = (LeidenDetector if community_algorithm == 'leiden' 
                      else SLLPADetector)(self.gds, self.graph)
            community_results = detector.process()
            
            # 4. 生成社区摘要
            summarizer = CommunitySummarizerFactory.create_summarizer(
                community_algorithm,
                self.graph
            )
            summaries = summarizer.process_communities()
            self._display_results_table(
                "社区摘要结果",
                {"生成的摘要数量": len(summaries) if summaries else 0}
            )
            
            self.console.print("[green]索引和社区构建完成[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]索引和社区构建失败: {str(e)}[/red]")
            return False

    def process_all(self):
        """执行完整的处理流程"""
        try:
            # 显示开始面板
            start_text = Text("开始知识图谱处理流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 1. 构建基础图谱
            if not self.build_base_graph():
                raise Exception("基础图谱构建失败")
            
            # 2. 构建索引和社区
            if not self.build_index_and_communities():
                raise Exception("索引和社区构建失败")
            
            # 显示完成面板
            success_text = Text("知识图谱处理流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
        except Exception as e:
            error_text = Text(f"处理过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise

if __name__ == "__main__":
    try:
        processor = KnowledgeGraphProcessor()
        processor.process_all()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")
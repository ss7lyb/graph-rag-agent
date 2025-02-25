from dotenv import load_dotenv
from typing import Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import time
import os
import psutil

from model.get_models import get_llm_model, get_embeddings_model
from config.prompt import (
    system_template_build_graph,
    human_template_build_graph
)
from config.settings import (
    entity_types,
    relationship_types,
    theme,
    FILES_DIR,
    CHUNK_SIZE,
    OVERLAP
)
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from graph.struct_builder import GraphStructureBuilder
from graph.entity_extractor import EntityRelationExtractor
from graph.graph_writer import GraphWriter
from langchain_community.graphs import Neo4jGraph

import shutup
shutup.please()

class KnowledgeGraphBuilder:
    """
    知识图谱构建器，负责图谱的基础构建流程。
    
    主要功能包括：
    1. 文件读取和解析
    2. 文本分块
    3. 实体和关系抽取
    4. 构建基础图结构
    5. 写入数据库
    """
    
    def __init__(self):
        """初始化知识图谱构建器"""
        # 初始化终端界面
        self.console = Console()
        self.file_contents = []
        
        # 加载环境变量
        load_dotenv()
        
        # 添加计时器
        self.start_time = None
        self.end_time = None
        
        # 阶段性能统计
        self.performance_stats = {
            "初始化": 0,
            "文件读取": 0,
            "文本分块": 0,
            "图结构构建": 0,
            "实体抽取": 0,
            "写入数据库": 0
        }
        
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
        init_start = time.time()
        
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=4)
            
            # 初始化模型
            self.llm = get_llm_model()
            self.embeddings = get_embeddings_model()
            progress.advance(task)
            
            # 初始化图数据库连接
            self.graph = Neo4jGraph()
            progress.advance(task)
            
            # 初始化文本处理器
            self.file_reader = FileReader(FILES_DIR)
            self.chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)
            progress.advance(task)
            
            # 初始化图谱构建器 - 动态调整批处理大小和并行度
            max_workers = os.cpu_count() or 4  # 如果无法确定CPU核心数，默认使用4
            # 根据系统内存动态调整批处理大小
            total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            optimal_batch_size = min(100, max(20, int(total_memory_gb * 5)))  # 根据可用内存估算
            
            self.struct_builder = GraphStructureBuilder(batch_size=optimal_batch_size)
            self.entity_extractor = EntityRelationExtractor(
                self.llm,
                system_template_build_graph,
                human_template_build_graph,
                entity_types,
                relationship_types,
                max_workers=max_workers,
                batch_size=5  # LLM批处理大小保持小一些以确保质量
            )
            
            # 输出优化参数
            self.console.print(f"[blue]并行处理线程数: {max_workers}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {optimal_batch_size}[/blue]")
            
            progress.advance(task)
        
        self.performance_stats["初始化"] = time.time() - init_start

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
        
    def _format_time(self, seconds: float) -> str:
        """格式化时间为小时:分钟:秒.毫秒"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_base_graph(self) -> List:
        """
        构建基础知识图谱
        
        Returns:
            List: 处理后的文件内容列表，包含文件名、原文、分块和处理结果
        """
        self._display_stage_header("构建基础知识图谱")
        
        try:
            # 1. 读取文件
            read_start = time.time()
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
            
            self.performance_stats["文件读取"] = time.time() - read_start
            
            # 2. 文本分块
            chunk_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]文本分块...", total=len(self.file_contents))
                for file_content in self.file_contents:
                    chunks = self.chunker.chunk_text(file_content[1])
                    file_content.append(chunks)
                    progress.advance(task)
            
            self.performance_stats["文本分块"] = time.time() - chunk_start
            
            # 显示分块统计
            total_chunks = sum(len(file_content[2]) for file_content in self.file_contents)
            avg_chunk_size = sum(len(''.join(chunk)) for file_content in self.file_contents for chunk in file_content[2]) / total_chunks if total_chunks else 0
            
            self.console.print(f"[blue]共生成 {total_chunks} 个文本块，平均每块 {avg_chunk_size:.1f} 字符[/blue]")
            
            # 3. 构建图结构
            struct_start = time.time()
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
                
                # 创建Chunk节点和关系 - 优化：使用并行处理大文件
                for file_content in self.file_contents:
                    # 根据chunks数量选择处理方法
                    chunks = file_content[2]
                    if len(chunks) > 100:
                        # 对于大文件使用并行处理
                        result = self.struct_builder.parallel_process_chunks(
                            file_content[0],
                            chunks,
                            max_workers=os.cpu_count() or 4
                        )
                    else:
                        # 对于小文件使用标准批处理
                        result = self.struct_builder.create_relation_between_chunks(
                            file_content[0],
                            chunks
                        )
                    file_content.append(result)
                progress.advance(task)
                progress.advance(task)
            
            self.performance_stats["图结构构建"] = time.time() - struct_start
            
            # 4. 提取实体和关系
            extract_start = time.time()
            with self._create_progress() as progress:
                total_chunks = sum(len(file_content[2]) for file_content in self.file_contents)
                task = progress.add_task("[cyan]提取实体和关系...", total=total_chunks)
                
                def progress_callback(chunk_index):
                    progress.advance(task)
                
                # 根据数据集大小选择处理方法
                if total_chunks > 100:
                    # 对于大型数据集使用批处理模式
                    self.file_contents = self.entity_extractor.process_chunks_batch(
                        self.file_contents,
                        progress_callback
                    )
                else:
                    # 对于小型数据集使用标准并行处理
                    self.file_contents = self.entity_extractor.process_chunks(
                        self.file_contents,
                        progress_callback
                    )
            
            self.performance_stats["实体抽取"] = time.time() - extract_start
            
            # 输出缓存统计
            cache_hits = getattr(self.entity_extractor, 'cache_hits', 0)
            cache_misses = getattr(self.entity_extractor, 'cache_misses', 0)
            total_requests = cache_hits + cache_misses
            cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            self.console.print(f"[blue]LLM调用缓存命中率: {cache_rate:.1f}% ({cache_hits}/{total_requests})[/blue]")
            
            # 5. 写入数据库
            write_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]写入数据库...", total=1)
                # 使用优化的GraphWriter
                graph_writer = GraphWriter(
                    self.graph, 
                    batch_size=50,
                    max_workers=os.cpu_count() or 4
                )
                graph_writer.process_and_write_graph_documents(self.file_contents)
                progress.update(task, completed=1)
            
            self.performance_stats["写入数据库"] = time.time() - write_start
            
            self.console.print("[green]基础知识图谱构建完成[/green]")
            
            # 显示性能统计
            performance_table = Table(title="性能统计")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")
            
            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)
            
            return self.file_contents
            
        except Exception as e:
            self.console.print(f"[red]基础图谱构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行知识图谱构建流程"""
        try:
            # 记录开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板
            start_text = Text("开始知识图谱构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 构建基础图谱
            result = self.build_base_graph()
            
            # 记录结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 显示完成面板
            success_text = Text("知识图谱构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return result
            
        except Exception as e:
            # 记录结束时间（即使出错）
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            error_text = Text(f"构建过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise

if __name__ == "__main__":
    try:
        builder = KnowledgeGraphBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")
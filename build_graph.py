from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from model.get_models import get_llm_model
from config.prompt import system_template_build_graph, human_template_build_graph
from config.settings import entity_types, relationship_types, theme, FILES_DIR, CHUNK_SIZE, OVERLAP
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from graph.struct_builder import GraphStructureBuilder
from graph.entity_extractor import EntityRelationExtractor
from graph.graph_writer import GraphWriter

import shutup
shutup.please()

class KnowledgeGraphBuilder:
    def __init__(self):
        self.console = Console()
        self.file_contents = []
        
    def create_progress(self):
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        )
    
    def read_files(self):
        """读取文件并显示基本信息"""
        with self.create_progress() as progress:
            task = progress.add_task("[cyan]Reading files...", total=1)
            fr = FileReader(FILES_DIR)
            self.file_contents = fr.read_txt_files()
            progress.update(task, completed=1)
            
            # 显示文件信息
            table = Table(show_header=True)
            table.add_column("File Name")
            table.add_column("Content Length", justify="right")
            for file_name, content in self.file_contents:
                table.add_row(file_name, str(len(content)))
            self.console.print(table)
    
    def chunk_texts(self):
        """文本分块处理"""
        with self.create_progress() as progress:
            task = progress.add_task("[cyan]Chunking texts...", total=len(self.file_contents))
            chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)
            
            for file_content in self.file_contents:
                chunks = chunker.chunk_text(file_content[1])
                file_content.append(chunks)
                progress.advance(task)
            
            # 显示分块结果
            table = Table(show_header=True)
            table.add_column("File Name")
            table.add_column("Chunks", justify="right")
            table.add_column("Total Tokens", justify="right")
            for file_content in self.file_contents:
                total_tokens = sum(len(chunk) for chunk in file_content[2])
                table.add_row(file_content[0], str(len(file_content[2])), str(total_tokens))
            self.console.print(table)
    
    def build_graph_structure(self):
        """构建图数据库结构"""
        with self.create_progress() as progress:
            task = progress.add_task("[cyan]Building graph structure...", total=3)
            
            struct_builder = GraphStructureBuilder()
            self.graph = struct_builder.graph
            progress.advance(task)
            
            # 清空并创建Document节点
            struct_builder.clear_database()
            for file_content in self.file_contents:
                struct_builder.create_document(
                    type="local",
                    uri=str(FILES_DIR),
                    file_name=file_content[0],
                    domain=theme
                )
            progress.advance(task)
            
            # 创建Chunk节点和关系
            for file_content in self.file_contents:
                result = struct_builder.create_relation_between_chunks(
                    file_content[0],
                    file_content[2]
                )
                file_content.append(result)
            progress.advance(task)
    
    def extract_entities(self):
        """使用LLM提取实体和关系"""
        with self.create_progress() as progress:
            self.console.print("\n[cyan]Initializing LLM model...")
            llm = get_llm_model()
            
            extractor = EntityRelationExtractor(
                llm=llm,
                system_template=system_template_build_graph,
                human_template=human_template_build_graph,
                entity_types=entity_types,
                relationship_types=relationship_types
            )
            
            total_chunks = sum(len(file_content[2]) for file_content in self.file_contents)
            task = progress.add_task(
                "[cyan]Extracting entities and relationships...",
                total=total_chunks
            )
            
            def progress_callback(chunk_index):
                progress.advance(task)
            
            self.file_contents = extractor.process_chunks(self.file_contents, progress_callback)
    
    def write_to_neo4j(self):
        """写入Neo4j数据库"""
        with self.create_progress() as progress:
            task = progress.add_task("[cyan]Writing to Neo4j...", total=1)
            graph_writer = GraphWriter(self.graph)
            graph_writer.process_and_write_graph_documents(self.file_contents)
            progress.update(task, completed=1)
    
    def build(self):
        """执行完整的知识图谱构建流程"""
        try:
            load_dotenv()
            self.console.print("[cyan]Starting knowledge graph construction...[/cyan]")
            
            self.read_files()
            self.chunk_texts()
            self.build_graph_structure()
            self.extract_entities()
            self.write_to_neo4j()
            
            self.console.print("[green]Knowledge Graph Building Complete![/green]")
            
        except KeyboardInterrupt:
            self.console.print("\n[red]Process interrupted by user[/red]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    builder.build()
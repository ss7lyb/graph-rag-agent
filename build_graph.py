from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.table import Table

from model.get_models import get_llm_model
from config.prompt import system_template, human_template
from config.settings import entity_types, relationship_types, theme, FILES_DIR, CHUNK_SIZE, OVERLAP
from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from graph.struct_builder import GraphStructureBuilder
from graph.entity_extractor import EntityRelationExtractor
from graph.graph_writer import GraphWriter

import shutup
shutup.please()

# 初始化Rich控制台
console = Console()

def create_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

def main():
    # 显示启动标题
    console.print(Panel.fit(
        Text("Knowledge Graph Builder", style="bold magenta"),
        border_style="bright_blue"
    ))
    
    load_dotenv()
    
    # 读入文件------------------------------------------------------------------
    with create_progress() as progress:
        file_task = progress.add_task("[cyan]Reading files...", total=1)
        fr = FileReader(FILES_DIR)
        file_contents = fr.read_txt_files()
        progress.update(file_task, completed=1)
        
        # 显示文件信息表格
        table = Table(title="Files Found", show_header=True, header_style="bold magenta")
        table.add_column("File Name", style="dim")
        table.add_column("Content Length", justify="right")
        for file_name, content in file_contents:
            table.add_row(file_name, str(len(content)))
        console.print(table)

    # 文本分块----------------------------------------------------------------------
    with create_progress() as progress:
        chunk_task = progress.add_task("[cyan]Chunking texts...", total=len(file_contents))
        chunker = ChineseTextChunker(CHUNK_SIZE, OVERLAP)
        
        for file_content in file_contents:
            chunks = chunker.chunk_text(file_content[1])
            file_content.append(chunks)
            progress.advance(chunk_task)
        
        # 显示分块结果
        table = Table(title="Chunking Results", show_header=True, header_style="bold magenta")
        table.add_column("File Name", style="dim")
        table.add_column("Chunks Count", justify="right")
        table.add_column("Total Tokens", justify="right")
        for file_content in file_contents:
            total_tokens = sum(len(chunk) for chunk in file_content[2])
            table.add_row(file_content[0], str(len(file_content[2])), str(total_tokens))
        console.print(table)

    # Neo4j图结构创建----------------------------------------------
    with create_progress() as progress:
        graph_task = progress.add_task("[cyan]Building graph structure...", total=3)
        
        # 实例化 GraphStructureBuilder
        struct_builder = GraphStructureBuilder()
        graph = struct_builder.graph
        progress.advance(graph_task)
        
        # 清空数据库并创建Document节点
        struct_builder.clear_database()
        for file_content in file_contents:
            struct_builder.create_document(
                type="local",
                uri=str(FILES_DIR),
                file_name=file_content[0],
                domain=theme
            )
        progress.advance(graph_task)
        
        # 创建Chunk节点和关系
        for file_content in file_contents:
            result = struct_builder.create_relation_between_chunks(
                file_content[0],
                file_content[2]
            )
            file_content.append(result)
        progress.advance(graph_task)

    # LLM实体关系提取------------------------------------
    with create_progress() as progress:
        console.print("\n[bold cyan]Initializing LLM model...[/bold cyan]")
        llm = get_llm_model()
        
        extractor = EntityRelationExtractor(
            llm=llm,
            system_template=system_template,
            human_template=human_template,
            entity_types=entity_types,
            relationship_types=relationship_types
        )
        
        total_chunks = sum(len(file_content[2]) for file_content in file_contents)
        extract_task = progress.add_task(
            "[cyan]Extracting entities and relationships...",
            total=total_chunks
        )
        
        def progress_callback(chunk_index):
            progress.advance(extract_task)
        
        # 添加进度回调
        file_contents = extractor.process_chunks(file_contents, progress_callback)

    # Neo4j写入-------------------------------------------------------
    with create_progress() as progress:
        write_task = progress.add_task("[cyan]Writing to Neo4j...", total=1)
        graph_writer = GraphWriter(graph)
        graph_writer.process_and_write_graph_documents(file_contents)
        progress.update(write_task, completed=1)

    # 完成提示
    console.print(Panel.fit(
        Text("Knowledge Graph Building Complete!", style="bold green"),
        border_style="bright_blue"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Process interrupted by user[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
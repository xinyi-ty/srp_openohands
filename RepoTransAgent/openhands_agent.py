import logging
import yaml
import shutil
from pathlib import Path
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

logger = logging.getLogger(__name__)


def load_llm_config(config_path: str = None) -> dict:
    """从 llm_config.yaml 文件加载 LLM 配置。"""
    if config_path is None:
        config_path = Path(__file__).parent / "llm_config.yaml"

    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    llm_config = config.get("llm", {})
    return {
        "model": llm_config.get("model"),
        "api_key": llm_config.get("api_key"),
        "base_url": llm_config.get("base_url"),
    }


def cleanup_source_files(target_dir: Path, source_extensions: list, target_extensions: list):
    """
    删除目标目录中所有源语言文件（如果存在对应的目标语言文件）。
    如果没有对应目标文件，则保留并记录警告。
    """
    deleted_count = 0
    for ext in source_extensions:
        for file_path in target_dir.rglob(f"*{ext}"):
            base_name = file_path.stem
            parent = file_path.parent
            found = False
            for t_ext in target_extensions:
                candidate = parent / f"{base_name}{t_ext}"
                if candidate.exists():
                    found = True
                    break
            if found:
                file_path.unlink()
                deleted_count += 1
                logger.info(f"已删除源文件: {file_path}")
            else:
                logger.warning(f"未找到对应的目标文件，保留源文件: {file_path}")
    logger.info(f"共删除 {deleted_count} 个源文件")


def run_translation(
        project_name: str,
        source_language: str,
        target_language: str,
        model_name: str,
        max_iterations: int,
        source_path: str,
        target_path: str,
) -> None:
    """
    使用 OpenHands 将项目从 source_language 翻译为 target_language。
    """
    # 1. 准备目标目录
    target_dir = Path(target_path)
    if target_dir.exists():
        logger.info(f"清空已存在的目标目录: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. 复制源项目
    source_dir = Path(source_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"源项目目录不存在: {source_dir}")

    logger.info(f"复制源项目 {source_dir} -> {target_dir}")
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    # 3. 加载 LLM 配置
    llm_config = load_llm_config()

    # 4. 初始化 LLM
    llm = LLM(
        model=model_name,
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
    )

    # 5. 注册工具
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]

    # 6. 创建 Agent
    agent = Agent(llm=llm, tools=tools)

    # 7. 强化后的任务指令（明确要求删除源文件）
    instruction = f"""You are an expert in code migration and translation.
Your task is to translate the project '{project_name}' from {source_language} to {target_language}.

The source code has been copied to the current working directory: {target_dir}
You have full access to read, edit, create, and delete files inside this directory.

**Critical rules for file management:**
1. For every source code file (extension specific to {source_language}), you MUST create a corresponding translated file with the **same base name** but the appropriate extension for {target_language}.
   Example: `src/main.cpp` (C++) → `src/main.py` (Python)
2. **After successfully creating the translated file, you MUST delete the original source file.** Do not leave any {source_language} files in the directory.
3. Do not mix {source_language} and {target_language} code in the same file. The translated file must contain only valid {target_language} code.
4. Preserve the directory structure exactly as in the source project.

Only after completing these steps should you begin translating individual files.

Workflow:
1. Analyze the project structure using `find`, `ls`, or `tree`.
2. For each source file:
   - Locate all relevant files written in {source_language}.
   - Read its content.
   - Translate it to {target_language}.
   - Write the translation to a new file with the same relative path but with the appropriate extension (e.g., .py for Python, .java for Java, .cpp for C++).
   - **Delete the original source file.**
3. For configuration files (CMakeLists.txt, Makefile, pom.xml, package.json, etc.), either translate them appropriately or delete them if not needed.
4. No extra modifications shall be made to files irrelevant to {source_language} and {target_language} languages.
5. Run the test suite using the appropriate command (e.g., `pytest`, `npm test`, `mvn test`). If tests are missing, you may skip.
6. If any test fails, analyze the error, fix the translated files, and re-run tests.
7. The task is complete when all tests pass (or after a reasonable effort).

You have a maximum of {max_iterations} steps.
"""

    # 8. 创建会话并运行
    conversation = Conversation(agent=agent, workspace=str(target_dir))
    conversation.send_message(instruction)
    conversation.run()

    # 9. 后处理：根据语言类型自动清理残留的源文件
    # 定义源语言和目标语言的文件扩展名（可根据需要扩展）
    language_extensions = {
        "c++": (['.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.hxx'], ['.py']),
        "python": (['.py'], ['.java', '.cpp']),
        "java": (['.java'], ['.py', '.cpp']),
        "javascript": (['.js', '.jsx'], ['.py']),
    }
    src_key = source_language.lower()
    # 直接获取扩展名元组，避免未使用变量的警告
    lang_entry = language_extensions.get(src_key)
    if lang_entry:
        source_extensions, target_extensions = lang_entry
    else:
        source_extensions = ['.c', '.cpp', '.h', '.py', '.java', '.js']
        target_extensions = ['.py'] if target_language.lower() == 'python' else ['.java']

    cleanup_source_files(target_dir, source_extensions, target_extensions)

    # 10. 可选：生成映射文件 MAPPING.txt
    mapping_file = target_dir / "MAPPING.txt"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write(f"# Translation mapping for project {project_name}\n")
        f.write(f"# Source: {source_language} -> Target: {target_language}\n\n")
        for t_ext in target_extensions:
            for target_file in target_dir.rglob(f"*{t_ext}"):
                f.write(f"{target_file.relative_to(target_dir)}\n")
        f.write("\nNote: Original source files have been deleted.\n")
    logger.info(f"已生成映射文件: {mapping_file}")

    logger.info(f"翻译任务完成: {project_name} -> {target_dir}")
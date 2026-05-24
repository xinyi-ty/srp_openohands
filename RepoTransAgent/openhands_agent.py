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
    返回一个列表，每个元素为 (源文件相对路径, 目标文件相对路径) 的元组。
    """
    deleted_count = 0
    mappings = []

    for ext in source_extensions:
        for file_path in target_dir.rglob(f"*{ext}"):
            base_name = file_path.stem
            parent = file_path.parent
            found = False
            target_file = None
            for t_ext in target_extensions:
                candidate = parent / f"{base_name}{t_ext}"
                if candidate.exists():
                    found = True
                    target_file = candidate
                    break
            if found:
                src_rel = file_path.relative_to(target_dir)
                tgt_rel = target_file.relative_to(target_dir)
                mappings.append((str(src_rel), str(tgt_rel)))

                file_path.unlink()
                deleted_count += 1
                logger.info(f"已删除源文件: {file_path}")
            else:
                logger.warning(f"未找到对应的目标文件，保留源文件: {file_path}")
    logger.info(f"共删除 {deleted_count} 个源文件")
    return mappings


def check_missing_files(source_dir: Path, target_dir: Path, source_extensions: list, target_extensions: list):
    """
    检查原始源项目中有哪些文件没有被翻译。
    返回缺失文件的相对路径列表。
    """
    missing = []
    for ext in source_extensions:
        for src_file in source_dir.rglob(f"*{ext}"):
            rel_path = src_file.relative_to(source_dir)
            base = src_file.stem
            found = False
            for t_ext in target_extensions:
                tgt_file = target_dir / rel_path.with_suffix(t_ext)
                if tgt_file.exists():
                    found = True
                    break
            if not found:
                missing.append(rel_path)
    return missing


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
        provider="openai",   # 显式指定 OpenAI 兼容模式
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 5. 注册工具
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]

    # 6. 创建 Agent
    agent = Agent(llm=llm, tools=tools)

    # 7. 优化后的任务指令（先翻译所有文件，再统一测试）
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

**Workflow (follow strictly):**
1. Analyze the project structure using `find`, `ls`, or `tree`.
2. **FIRST, translate ALL source files** (one by one) without stopping to run any compile or test commands.
   - For each source file: read → translate → write target file → delete source file.
   - Do NOT run any compile/test commands during this phase.
3. After ALL files have been translated, **then** run the test suite once using the appropriate command (e.g., `pytest`, `npm test`, `mvn test`). If tests are missing, you may skip.
4. If tests fail, analyze the errors and fix the relevant translated files.
5. Re-run the tests after fixing. Repeat until all tests pass or you run out of steps.

**Error handling during testing:**
- If a test command fails, carefully read the error output.
- Identify which file(s) caused the failure, fix them, and re-run the test command.
- Do not go back to translating files that are already done unless fixing errors.

You have a maximum of {max_iterations} steps.
The task is complete when all tests pass (or after a reasonable effort if no tests exist).
"""

    # 8. 创建会话并运行
    conversation = Conversation(agent=agent, workspace=str(target_dir))
    conversation.send_message(instruction)
    conversation.run()

    # 9. 后处理：根据语言类型自动清理残留的源文件，并收集映射关系
    language_extensions = {
        "c++": (['.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.hxx'], ['.py']),
        "python": (['.py'], ['.java', '.cpp']),
        "java": (['.java'], ['.py', '.cpp']),
        "javascript": (['.js', '.jsx'], ['.py']),
    }
    src_key = source_language.lower()
    lang_entry = language_extensions.get(src_key)
    if lang_entry:
        source_extensions, target_extensions = lang_entry
    else:
        source_extensions = ['.c', '.cpp', '.h', '.py', '.java', '.js']
        target_extensions = ['.py'] if target_language.lower() == 'python' else ['.java']

    mappings = cleanup_source_files(target_dir, source_extensions, target_extensions)

    # 10. 生成映射文件 MAPPING.txt（格式：源文件 -> 目标文件）
    mapping_file = target_dir / "MAPPING.txt"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write(f"# Translation mapping for project: {project_name}\n")
        f.write(f"# Source language: {source_language} → Target language: {target_language}\n")
        f.write("# Format: source_file -> target_file\n\n")
        if mappings:
            for src, tgt in mappings:
                f.write(f"{src} -> {tgt}\n")
        else:
            f.write("# No source files were deleted (possible incomplete translation).\n")
            for t_ext in target_extensions:
                for target_file in target_dir.rglob(f"*{t_ext}"):
                    f.write(f"# (inferred) {target_file.stem}.* -> {target_file.relative_to(target_dir)}\n")
        f.write("\nNote: Original source files have been deleted where corresponding target files exist.\n")
    logger.info(f"已生成映射文件: {mapping_file}")

    # 11. 检查缺失的翻译文件（兜底检测）
    missing = check_missing_files(source_dir, target_dir, source_extensions, target_extensions)
    if missing:
        logger.warning(f"发现 {len(missing)} 个未翻译的源文件，可能因步数不足导致漏译:")
        for m in missing[:10]:   # 只显示前10个
            logger.warning(f"  - {m}")
        if len(missing) > 10:
            logger.warning(f"  ... 以及 {len(missing)-10} 个更多文件")
        # 保存完整列表到文件
        missing_file = target_dir / "MISSING_FILES.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write(f"# 以下 {len(missing)} 个源文件在翻译后未能生成对应的目标文件\n")
            f.write("# 可能原因：Agent 步数不足、翻译失败、或文件未被识别\n\n")
            for m in missing:
                f.write(f"{m}\n")
        logger.info(f"未翻译文件列表已保存到 {missing_file}")
    else:
        logger.info("所有源文件均已翻译，无缺失。")

    logger.info(f"翻译任务完成: {project_name} -> {target_dir}")
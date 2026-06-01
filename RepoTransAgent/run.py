import argparse
import logging
import sys
from RepoTransAgent.openhands_agent import run_translation

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 OpenHands 运行单个项目的代码翻译（源/目标目录分离）。"
    )
    parser.add_argument("--project_name", required=True, help="项目名称（仅用于日志和标识）")
    parser.add_argument("--source_language", required=True, help="源语言，如 Python")
    parser.add_argument("--target_language", required=True, help="目标语言，如 Java")
    parser.add_argument(
        "--model_name",
        default="deepseek-v4-flash",
        help="LLM 模型名称",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=50, help="Agent 最大执行步数（增大以避免漏译）"
    )
    parser.add_argument(
        "--source_path",
        required=True,
        help="源项目的绝对路径（例如 D:/workspace/projects/source_projects/my_project）",
    )
    parser.add_argument(
        "--target_path",
        required=True,
        help="目标项目输出路径（例如 D:/workspace/projects/target_projects/my_project）",
    )

    args = parser.parse_args()

    run_translation(
        project_name=args.project_name,
        source_language=args.source_language,
        target_language=args.target_language,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        source_path=args.source_path,
        target_path=args.target_path,
    )
from pathlib import Path
from typing import List

from app.ingestion.detectors import is_supported_file


def load_single_file(file_path: str) -> str:
    """
    校验并返回单个文件的绝对路径。

    只负责：
    - 路径存在性检查
    - 是否为文件检查
    - 是否为支持类型检查

    不负责：
    - 读取文件内容
    - 解析文档
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是有效文件: {file_path}")

    if not is_supported_file(str(path)):
        raise ValueError(f"暂不支持的文件类型: {path.suffix.lower()}")

    return str(path)


def load_file_paths_from_folder(folder_path: str, recursive: bool = True) -> List[str]:
    """
    从本地文件夹中加载所有支持的文件路径。

    参数：
    - folder_path: 文件夹路径
    - recursive: 是否递归扫描子目录

    返回：
    - 支持类型文件的绝对路径列表（已排序）
    """
    folder = Path(folder_path).expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"不是有效文件夹: {folder_path}")

    pattern = "**/*" if recursive else "*"
    file_paths = []

    for path in folder.glob(pattern):
        if path.is_file() and is_supported_file(str(path)):
            file_paths.append(str(path.resolve()))

    return sorted(file_paths)


def load_multiple_files(file_paths: List[str]) -> List[str]:
    """
    校验多个文件路径，返回合法文件的绝对路径列表。
    """
    valid_files = []
    for file_path in file_paths:
        valid_files.append(load_single_file(file_path))
    return valid_files


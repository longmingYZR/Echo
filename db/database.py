"""
数据库连接工厂（单例模式）
"""

import os
import sqlite3
import threading
from .schema import DB_PATH, init_db


_local = threading.local()


def get_db() -> sqlite3.Connection:
    """获取当前线程的数据库连接。首次调用自动建表并初始化。"""
    if not hasattr(_local, "db") or _local.db is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.db.row_factory = sqlite3.Row
        init_db(_local.db)
    return _local.db


def close_db():
    """关闭当前线程的数据库连接。"""
    if hasattr(_local, "db") and _local.db is not None:
        _local.db.close()
        _local.db = None

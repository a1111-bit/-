"""
用户账户与问诊记录存储（本地 SQLite 数据库）。

只用 Python 标准库实现，无需安装任何额外依赖：
- 密码用 PBKDF2-SHA256 加盐哈希存储（不存明文）；
- 每个用户的问诊记录（含聊天历史与诊断报告）按用户名隔离保存。

数据库文件放在 outputs/users.db（outputs/ 已被 .gitignore 忽略，不会误提交）。
"""

import os
import json
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime

from evoagentx_medical_config import OUTPUTS_DIR

# 数据库路径（模块级全局，单测可临时改指向临时文件）
DB_PATH = os.path.join(OUTPUTS_DIR, "users.db")

_PBKDF2_ITERATIONS = 200_000


def _connect() -> sqlite3.Connection:
    """打开数据库连接，并在首次使用时建好表（幂等）。每次操作独立连接，线程安全。"""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            name TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(username, name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(username)")
    return conn


def _hash_password(password: str, salt_hex=None):
    """返回 (盐的十六进制, 哈希的十六进制)。不传盐则随机生成新盐。"""
    if salt_hex is None:
        salt = secrets.token_bytes(16)
    else:
        salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def _valid_username(username: str) -> bool:
    if not (3 <= len(username) <= 20):
        return False
    # isalnum() 对中文也返回 True，因此同时支持中文/字母/数字；另允许下划线
    return all(c.isalnum() or c == "_" for c in username)


def register(username: str, password: str) -> str:
    """注册新账户。成功返回空字符串，失败返回错误提示。"""
    username = (username or "").strip()
    password = password or ""
    if not _valid_username(username):
        return "用户名需为 3-20 位（可用中文、字母、数字或下划线）。"
    if len(password) < 6:
        return "密码至少 6 位。"

    salt_hex, hash_hex = _hash_password(password)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, hash_hex, salt_hex, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return ""
    except sqlite3.IntegrityError:
        return "用户名已存在，请换一个。"
    finally:
        conn.close()


def login(username: str, password: str) -> str:
    """校验账户。成功返回空字符串，失败返回错误提示（不区分用户名/密码哪个错）。"""
    username = (username or "").strip()
    password = password or ""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return "用户名或密码错误。"
    stored_hash, salt_hex = row
    _, computed_hash = _hash_password(password, salt_hex)
    # 常量时间比较，防时序攻击
    if not hmac.compare_digest(stored_hash, computed_hash):
        return "用户名或密码错误。"
    return ""


def save_conversation(username: str, name: str, state: dict) -> None:
    """把一次问诊的完整状态（聊天历史 + 分析报告）存到该用户名下；同名则覆盖更新。"""
    if not username or not name:
        return
    conn = _connect()
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state_json = json.dumps(state, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO conversations (username, name, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username, name)
            DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at
            """,
            (username, name, state_json, now),
        )
        conn.commit()
    finally:
        conn.close()


def load_conversation(username: str, name: str):
    """读取该用户名下某个问诊记录，返回状态字典；不存在或损坏返回 None。"""
    if not username or not name:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT state_json FROM conversations WHERE username = ? AND name = ?",
            (username, name),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def list_conversations(username: str):
    """返回该用户名下的所有问诊记录名（按更新时间倒序）。"""
    if not username:
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT name FROM conversations WHERE username = ? ORDER BY updated_at DESC, id DESC",
            (username,),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()

"""
用户账户与问诊记录存储（medical_auth.py）的单元测试。

只测本地 SQLite 逻辑，不联网、不调模型。测试会临时把数据库指向临时目录，
不会污染真实的 outputs/users.db。

运行方式（在 app 目录下）：
    python test_medical_auth.py
或：
    python -m unittest test_medical_auth -v
"""

import os
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

# 确保能 import 同目录下的模块（从任意目录运行都能找到 app 下的文件）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import medical_auth as auth


class TestMedicalAuth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = auth.DB_PATH
        # 测试全程指向临时库，避免写入真实数据
        auth.DB_PATH = os.path.join(self._tmp.name, "test_users.db")

    def tearDown(self):
        auth.DB_PATH = self._old_db
        self._tmp.cleanup()

    def test_register_and_login_success(self):
        self.assertEqual(auth.register("alice", "secret123"), "")
        self.assertEqual(auth.login("alice", "secret123"), "")

    def test_login_wrong_password(self):
        auth.register("alice", "secret123")
        self.assertNotEqual(auth.login("alice", "wrongpass"), "")

    def test_login_unknown_user(self):
        self.assertNotEqual(auth.login("nobody", "secret123"), "")

    def test_register_duplicate(self):
        self.assertEqual(auth.register("bob", "secret123"), "")
        self.assertNotEqual(auth.register("bob", "another456"), "")

    def test_register_short_password(self):
        self.assertNotEqual(auth.register("bob", "12345"), "")

    def test_register_invalid_username(self):
        self.assertNotEqual(auth.register("a!", "secret123"), "")
        self.assertNotEqual(auth.register("ab", "secret123"), "")

    def test_password_not_stored_in_plaintext(self):
        """密码必须加盐哈希存储，不能是明文。"""
        auth.register("carol", "secret123")
        conn = sqlite3.connect(auth.DB_PATH)
        try:
            password_hash, salt = conn.execute(
                "SELECT password_hash, salt FROM users WHERE username = 'carol'"
            ).fetchone()
        finally:
            conn.close()
        self.assertNotEqual(password_hash, "secret123")
        self.assertNotIn("secret123", password_hash)
        self.assertTrue(salt)

    def test_save_list_load_roundtrip(self):
        """保存→列出→加载应能完整还原会话字典。"""
        auth.register("dave", "secret123")
        state = {"stage": "confirm", "history": [{"role": "user", "content": "头痛"}]}
        auth.save_conversation("dave", "张三-头痛", state)

        self.assertIn("张三-头痛", auth.list_conversations("dave"))
        loaded = auth.load_conversation("dave", "张三-头痛")
        self.assertEqual(loaded, state)

    def test_save_overwrites_same_name(self):
        """同名再次保存应更新而非新增重复记录。"""
        auth.register("eve", "secret123")
        auth.save_conversation("eve", "病例A", {"stage": "intake"})
        auth.save_conversation("eve", "病例A", {"stage": "confirm"})
        names = auth.list_conversations("eve")
        self.assertEqual(names.count("病例A"), 1)
        self.assertEqual(auth.load_conversation("eve", "病例A"), {"stage": "confirm"})

    def test_users_are_isolated(self):
        """不同用户看不到彼此的记录。"""
        auth.register("u1", "secret123")
        auth.register("u2", "secret123")
        auth.save_conversation("u1", "只有u1的记录", {"stage": "intake"})
        self.assertIn("只有u1的记录", auth.list_conversations("u1"))
        self.assertNotIn("只有u1的记录", auth.list_conversations("u2"))


if __name__ == "__main__":
    unittest.main()

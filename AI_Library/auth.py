# auth.py
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Tuple
try:
    from user_manager import (
        create_user_record,
        delete_user_record,
        list_user_records,
        update_user_password_record,
        update_user_role_record,
    )
except ImportError:
    from .user_manager import (
        create_user_record,
        delete_user_record,
        list_user_records,
        update_user_password_record,
        update_user_role_record,
    )

class UserManager:
    """用户账户管理器"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()

    
    def init_database(self):
        """初始化用户数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 创建默认管理员账户
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role) 
            VALUES (?, ?, ?)
        ''', ('admin', self._hash_password('admin123'), 'admin'))
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str = 'user') -> bool:
        """创建新用户"""
        success, _ = create_user_record(self.db_path, username, password, role)
        return success

    def list_users(self) -> list:
        """返回所有用户，供管理员界面展示。"""
        return list_user_records(self.db_path)

    def add_user(self, username: str, password: str, role: str = "user") -> Tuple[bool, str]:
        """创建新用户并返回状态消息。"""
        return create_user_record(self.db_path, username, password, role)

    def delete_user_by_username(
        self,
        username: str,
        current_username: str = None,
        secondary_password: str = "",
        admin_secondary_password: str = "",
    ) -> Tuple[bool, str]:
        """按用户名删除用户，禁止删除当前登录用户。"""
        return delete_user_record(
            self.db_path,
            username,
            current_username,
            secondary_password,
            admin_secondary_password,
        )

    def update_user_role(self, username: str, role: str, current_username: str = None) -> Tuple[bool, str]:
        """修改用户权限，禁止修改当前登录用户自己的权限。"""
        return update_user_role_record(self.db_path, username, role, current_username)

    def update_user_password(
        self,
        username: str,
        new_password: str,
        secondary_password: str = "",
        admin_secondary_password: str = "",
    ) -> Tuple[bool, str]:
        """重置用户密码；管理员账号需要二级密码。"""
        return update_user_password_record(
            self.db_path,
            username,
            new_password,
            secondary_password,
            admin_secondary_password,
        )
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """用户认证"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, role FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, self._hash_password(password)))
        
        result = cursor.fetchone()

        if result:
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result[0]),
            )
            conn.commit()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2]
            }
        return None
    
    def create_session(self, user_id: int) -> str:
        """创建用户会话"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now().timestamp() + 24 * 3600  # 24小时过期
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, expires_at)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, expires_at))
        
        conn.commit()
        conn.close()
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[dict]:
        """验证会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.username, u.role 
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_id = ? AND s.expires_at > ?
        ''', (session_id, datetime.now().timestamp()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'role': result[2]
            }
        return None
    
    def delete_session(self, session_id: str):
        """删除会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()

import sys
import getpass
import sqlite3
import hashlib
import secrets
from datetime import datetime

VALID_ROLES = {"user", "admin"}


def hash_password(password):
    """生成与 auth.UserManager 一致的密码哈希。"""
    return hashlib.sha256(password.encode()).hexdigest()


def normalize_role(role):
    role = (role or "user").strip().lower()
    return role if role in VALID_ROLES else ""


def verify_secondary_password(secondary_password, admin_secondary_password):
    expected = str(admin_secondary_password or "")
    return bool(expected) and str(secondary_password or "") == expected


def list_user_records(db_path="users.db"):
    """返回用户列表，供 CLI 和 Web 管理界面复用。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, role, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "username": row[1],
            "role": row[2],
            "created_at": row[3],
            "last_login": row[4],
        }
        for row in rows
    ]


def create_user_record(db_path, username, password, role="user"):
    """创建用户；返回 (success, message)。"""
    username = (username or "").strip()
    role = normalize_role(role)
    if not username:
        return False, "用户名不能为空"
    if not password or len(password) < 6:
        return False, "密码至少需要6个字符"
    if not role:
        return False, "角色必须是 user 或 admin"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            """,
            (username, hash_password(password), role),
        )
        conn.commit()
        conn.close()
        return True, f"用户 '{username}' 创建成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    except Exception as e:
        return False, f"创建用户失败: {e}"


def delete_user_record(db_path, username, current_username=None, secondary_password="", admin_secondary_password=""):
    """删除用户并清理会话；禁止删除当前登录账号。"""
    username = (username or "").strip()
    current_username = (current_username or "").strip()
    if not username:
        return False, "请选择要删除的用户"
    if current_username and username == current_username:
        return False, "不能删除当前登录用户"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return False, f"用户 '{username}' 不存在"

        user_id, role = user
        if role == "admin" and not verify_secondary_password(secondary_password, admin_secondary_password):
            conn.close()
            return False, "删除管理员账户需要正确的二级密码"

        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True, f"用户 '{username}' 已删除"
    except Exception as e:
        return False, f"删除用户失败: {e}"


def update_user_role_record(db_path, username, role, current_username=None):
    """修改用户权限；禁止修改当前登录账号自己的权限。"""
    username = (username or "").strip()
    current_username = (current_username or "").strip()
    role = normalize_role(role)
    if not username:
        return False, "请选择要修改的用户"
    if not role:
        return False, "角色必须是 user 或 admin"
    if current_username and username == current_username:
        return False, "不能修改当前登录用户自己的权限"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not cursor.fetchone():
            conn.close()
            return False, f"用户 '{username}' 不存在"
        cursor.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        conn.commit()
        conn.close()
        return True, f"用户 '{username}' 权限已更新为 {role}"
    except Exception as e:
        return False, f"修改用户权限失败: {e}"


def update_user_password_record(db_path, username, new_password, secondary_password="", admin_secondary_password=""):
    """重置用户密码；管理员账户需要二级密码。"""
    username = (username or "").strip()
    if not username:
        return False, "请选择要修改密码的用户"
    if not new_password or len(new_password) < 6:
        return False, "新密码至少需要6个字符"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return False, f"用户 '{username}' 不存在"
        user_id, role = user
        if role == "admin" and not verify_secondary_password(secondary_password, admin_secondary_password):
            conn.close()
            return False, "修改管理员密码需要正确的二级密码"

        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), user_id),
        )
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True, f"用户 '{username}' 密码已修改"
    except Exception as e:
        return False, f"修改用户密码失败: {e}"


def print_usage():
    """打印使用说明"""
    print("用户账户管理工具")
    print("用法:")
    print("  python user_manager.py list              # 列出所有用户")
    print("  python user_manager.py add               # 添加新用户")
    print("  python user_manager.py delete <username> # 删除用户")
    print("  python user_manager.py reset <username>  # 重置用户密码")
    print("  python user_manager.py stats             # 显示统计信息")

def list_users():
    """列出所有用户"""
    try:
        users = list_user_records("users.db")
        
        if not users:
            print("❌ 没有找到用户")
            return
        
        print("\n📋 用户列表:")
        print("-" * 80)
        
        # 检查查询结果的长度
        try:
            print(f"{'ID':<3} {'用户名':<15} {'角色':<8} {'创建时间':<19} {'最后登录':<19}")
            print("-" * 80)
            for user in users:
                user_id = user["id"]
                username = user["username"]
                role = user["role"]
                created_at = user["created_at"]
                last_login = user["last_login"]
                last_login_str = last_login[:19] if last_login else "从未登录"
                created_at_str = created_at[:19] if created_at else "未知"
                print(f"{user_id:<3} {username:<15} {role:<8} {created_at_str:<19} {last_login_str:<19}")
        except Exception as e:
            print(f"检查查询长度失败{e}")
        print("-" * 80)
        print(f"总计: {len(users)} 个用户")
        
    except Exception as e:
        print(f"❌ 获取用户列表失败: {e}")

def add_user():
    """添加新用户"""
    print("\n👤 添加新用户")
    
    username = input("用户名: ").strip()
    if not username:
        print("❌ 用户名不能为空")
        return
    
    password = getpass.getpass("密码: ").strip()
    if len(password) < 6:
        print("❌ 密码至少需要6个字符")
        return
    
    confirm_password = getpass.getpass("确认密码: ").strip()
    if password != confirm_password:
        print("❌ 密码不匹配")
        return
    
    role = input("角色 (user/admin, 默认为user): ").strip()
    if not role:
        role = "user"
    
    if role not in ["user", "admin"]:
        print("❌ 角色必须是 user 或 admin")
        return
    

    
    success, message = create_user_record("users.db", username, password, role)
    print(("✅ " if success else "❌ ") + message)

def delete_user(username):
    """删除用户"""
    if not username:
        print("❌ 请提供用户名")
        return
    
    # 确认删除
    confirm = input(f"⚠️  确认删除用户 '{username}'？此操作不可恢复！(y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作取消")
        return
    
    success, message = delete_user_record("users.db", username)
    print(("✅ " if success else "❌ ") + message)

def reset_password(username):
    """重置用户密码"""
    if not username:
        print("❌ 请提供用户名")
        return
    
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        # 检查用户是否存在
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            print(f"❌ 用户 '{username}' 不存在")
            conn.close()
            return
        user_id = user[0]
        
        new_password = getpass.getpass("新密码: ").strip()
        if len(new_password) < 6:
            print("❌ 密码至少需要6个字符")
            conn.close()
            return
        
        confirm_password = getpass.getpass("确认新密码: ").strip()
        if new_password != confirm_password:
            print("❌ 密码不匹配")
            conn.close()
            return
        
        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            
        cursor.execute(
            'UPDATE users SET password_hash = ? WHERE id = ?',
            (password_hash, user_id)
            )
        
        conn.commit()
        conn.close()
        
        print(f"✅ 用户 '{username}' 的密码已重置")
        
    except Exception as e:
        print(f"❌ 重置密码失败: {e}")

def show_stats():
    """显示统计信息"""
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        stats = {}
        
        # 总用户数
        cursor.execute('SELECT COUNT(*) FROM users')
        stats['total_users'] = cursor.fetchone()[0]
        
        # 活跃用户数（假设都活跃）
        stats['active_users'] = stats['total_users']
        # 管理员数
        cursor.execute('SELECT COUNT(*) FROM users WHERE role = "admin"')
        stats['admin_count'] = cursor.fetchone()[0]
        
        # 活跃会话数（如果sessions表存在）
        try:
            cursor.execute('SELECT COUNT(*) FROM sessions WHERE expires_at > ?', 
                          (datetime.now().timestamp(),))
            stats['active_sessions'] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            stats['active_sessions'] = 0
        
        conn.close()
        
        print("\n📊 用户统计信息:")
        print(f"   总用户数: {stats.get('total_users', 0)}")
        print(f"   活跃用户: {stats.get('active_users', 0)}")
        print(f"   管理员数: {stats.get('admin_count', 0)}")
        print(f"   活跃会话: {stats.get('active_sessions', 0)}")
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_users()
    elif command == "add":
        add_user()
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ 请提供要删除的用户名")
            return
        delete_user(sys.argv[2])
    elif command == "reset":
        if len(sys.argv) < 3:
            print("❌ 请提供要重置密码的用户名")
            return
        reset_password(sys.argv[2])
    elif command == "stats":
        show_stats()
    elif command == "help":
        print_usage()
    else:
        print("❌ 未知命令")
        print_usage()

if __name__ == "__main__":
    main()

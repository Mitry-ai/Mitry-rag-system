# AI 智库本地知识库系统

这是一个基于本地大模型的知识库问答系统，支持 PDF/TXT 文档上传、向量化检索、RAG 问答、管理员文档管理、用户管理、审计日志和向量库同步维护。

## 功能概览

- 本地知识库问答：基于 Chroma 向量库和 Ollama 本地模型。
- 多模式问答：精准检索、综合模式、探索模式。
- 文档管理：管理员可上传、查看下载、删除文档。
- 向量库同步：删除文档后同步删除向量块，并更新问答链。
- 用户管理：管理员可查看用户、添加用户、删除用户、修改权限、修改密码。
- 权限控制：普通用户只能使用问答功能，管理员可访问文档管理和用户管理。
- 审计日志：记录问答 trace、检索信息、引用信息和状态。

## 目录结构

```text
AI_Library/
  ai_library.py          # Gradio Web 主程序
  auth.py                # 登录、会话、Web 用户管理封装
  user_manager.py        # 命令行用户管理脚本和用户数据操作函数
  config.py              # 系统配置和环境变量读取
  sync_vector_db.py      # documents 与 Chroma 向量库同步脚本
  pdf_optimizer.py       # PDF 解析处理
  network.py             # 联网工具能力
  audit.py               # 审计日志
  retrieval/             # 检索、上下文构建、引用、回答编排
  documents/             # 本地文档目录，不建议提交真实文件
  chroma_db_local/       # 本地向量库目录，不应提交
  logs/                  # 日志目录，不应提交
```

## 环境要求

- Python 3.10+
- Ollama
- 本地 Ollama 模型：
  - LLM：`deepseek-r1`
  - Embedding：`nomic-embed-text:latest`

安装 Ollama 模型：

```powershell
ollama pull deepseek-r1
ollama pull nomic-embed-text:latest
```

## Python 依赖安装

建议在虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

如果 PDF 解析只使用 PyMuPDF/PyPDF，`unstructured` 不是必须项；安装失败时可以先跳过。

## 配置说明

主要配置位于 `AI_Library/config.py`。

### 基础配置

```python
SYSTEM_CONFIG = {
    "max_concurrent_users": 1,
    "default_model": "deepseek-r1",
    "embedding_model": "nomic-embed-text:latest"
}
```

- `default_model`：问答使用的 Ollama LLM 模型。
- `embedding_model`：向量化使用的 Ollama embedding 模型。
- `max_concurrent_users`：问答并发上限。

注意：如果更换 `embedding_model`，通常需要重建向量数据库。

### 二级密码

管理员执行高风险操作时需要二级密码：

- 修改管理员账号密码
- 删除管理员账号

二级密码不再写死在代码中，而是从环境变量读取：

```powershell
$env:AI_ADMIN_SECONDARY_PASSWORD="123456"
```

启动 Web 前必须设置该环境变量。未设置时，管理员账号的高风险操作会被拒绝。

### 常用环境变量

```powershell
# Web 端口，默认 7860
$env:AI_WEB_PORT="7860"

# 管理员高风险操作二级密码
$env:AI_ADMIN_SECONDARY_PASSWORD="123456"

# 审计日志开关，默认开启
$env:AI_AUDIT_ENABLED="1"

# 是否把完整答案写入审计日志，默认关闭
$env:AI_AUDIT_INCLUDE_ANSWER="0"

# 检索策略：hybrid 或 dense
$env:AI_RETRIEVAL_STRATEGY="hybrid"
```

更多检索、rerank、拒答、query rewrite 参数可在 `config.py` 的 `RETRIEVAL_CONFIG` 中查看。

## 启动 Web 系统

建议在项目根目录执行：

```powershell
$env:AI_ADMIN_SECONDARY_PASSWORD="123456"
python AI_Library\ai_library.py
```

访问：

```text
http://localhost:7860
```

如端口被占用：

```powershell
$env:AI_WEB_PORT="7861"
python AI_Library\ai_library.py
```

默认管理员账号：

```text
username: admin
password: admin123
```

该默认账号仅用于本地演示。生产环境应初始化后立即修改密码，并使用更安全的密码哈希方案。

## Web 端使用说明

### 普通用户

普通用户登录后可以使用知识库问答功能。

### 管理员

管理员登录后会看到两个一级入口：

- 文档管理
- 用户管理

#### 文档管理

支持：

- 上传 PDF/TXT
- 查看文档列表
- 点击文件名下载文档
- 选择文档并删除

删除文档后，系统会同步删除 Chroma 中对应向量块，并重新更新问答链。

#### 用户管理

支持：

- 查看用户列表
- 添加用户，可选择普通用户或管理员
- 删除用户
- 修改用户权限
- 修改用户密码

安全限制：

- 普通用户不能访问用户管理。
- 后端会校验管理员身份，不只依赖前端隐藏。
- 禁止删除当前登录用户。
- 禁止修改当前登录用户自己的权限。
- 删除管理员账号需要二级密码。
- 修改管理员账号密码需要二级密码。

## 向量数据库同步脚本

脚本路径：

```text
AI_Library/sync_vector_db.py
```

建议进入 `AI_Library` 目录执行：

```powershell
cd AI_Library
```

查看统计：

```powershell
python sync_vector_db.py stats
```

同步删除：删除向量库中存在、但 `documents` 文件夹中已不存在的文件：

```powershell
python sync_vector_db.py sync-delete
```

干运行同步删除：

```powershell
python sync_vector_db.py sync-delete --dry-run
```

同步新增：把 `documents` 文件夹中新出现的 PDF/TXT 添加到向量库：

```powershell
python sync_vector_db.py sync-add
```

干运行同步新增：

```powershell
python sync_vector_db.py sync-add --dry-run
```

完整同步：先同步删除，再同步新增：

```powershell
python sync_vector_db.py full-sync
```

重建向量库：

```powershell
python sync_vector_db.py rebuild
```

跳过确认重建：

```powershell
python sync_vector_db.py rebuild --yes
```

注意：

- 执行同步脚本前需要确保 Ollama embedding 模型可用。
- 重建或写入向量库时，建议关闭正在运行的 Web 服务，避免 Chroma 文件被占用。
- 更换 `embedding_model` 后建议执行 `rebuild`。

## 用户管理脚本

脚本路径：

```text
AI_Library/user_manager.py
```

重要：该脚本使用当前工作目录下的 `users.db`。请先进入 `AI_Library` 目录再执行：

```powershell
cd AI_Library
```

查看帮助：

```powershell
python user_manager.py help
```

列出用户：

```powershell
python user_manager.py list
```

添加用户，按提示输入用户名、密码和角色：

```powershell
python user_manager.py add
```

删除用户：

```powershell
python user_manager.py delete <username>
```

重置用户密码：

```powershell
python user_manager.py reset <username>
```

查看用户统计：

```powershell
python user_manager.py stats
```

说明：

- Web 端支持修改用户权限。
- 命令行脚本当前主要支持用户列表、添加、删除、重置密码、统计。

## GitHub 提交建议

不要提交以下本地产物：

```gitignore
__pycache__/
*.pyc

AI_Library/users.db
AI_Library/chroma_db_local/
AI_Library/logs/
AI_Library/documents/*
!AI_Library/documents/.gitkeep

.env
*.log
```

原因：

- `users.db` 包含用户、密码哈希和 session。
- `chroma_db_local/` 是本地向量库，可能包含文档语义信息。
- `logs/` 可能包含问答内容和调试信息。
- `documents/` 中的真实 PDF/TXT 可能涉及版权或隐私。
- `__pycache__/`、`*.pyc` 是 Python 编译缓存。

如果希望保留空的 `documents` 目录，可以放置空文件：

```text
AI_Library/documents/.gitkeep
```

## 常见问题

### Ollama 连接失败

确认 Ollama 正在运行，并且模型已下载：

```powershell
ollama list
ollama pull deepseek-r1
ollama pull nomic-embed-text:latest
```

如果使用代理，确保本地地址不走代理：

```powershell
$env:NO_PROXY="localhost,127.0.0.1,::1"
```

### 端口被占用

默认端口是 `7860`，可使用：

```powershell
$env:AI_WEB_PORT="7861"
python AI_Library\ai_library.py
```

### 二级密码不生效

确认启动 Web 前已经设置：

```powershell
$env:AI_ADMIN_SECONDARY_PASSWORD="123456"
```

环境变量在 Python 进程启动时读取。修改环境变量后，需要重启 Web 服务。

### 更换模型后检索效果异常

如果更换的是 `default_model`，一般重启服务即可。

如果更换的是 `embedding_model`，建议重建向量库：

```powershell
cd AI_Library
python sync_vector_db.py rebuild --yes
```

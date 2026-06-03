# LLM Wiki Engine

纯后端 LLM 知识编译引擎 — **Compile-time Knowledge Synthesis**。

不同于传统 RAG（chunk → embedding → 检索原文），本引擎先将原始文档通过 LLM **两步编译** 为结构化 Markdown 知识单元（entity / concept / source summary），再基于高质量知识页面提供检索、问答、Agent 与反馈修正 API。

> 完整栈（UI + Case Service）请使用仓库根目录的 `docker-compose.yml` 或 `./start-dev.sh`。本文档侧重 **单独运行 engine** 或 API 集成。

## 核心能力

- **两步 CoT Ingest** — Analysis → Generation → FILE 块解析 → LLM 辅助页面合并 → SHA256 增量缓存 → 步骤级 checkpoint
- **Git 仓库同步** — 项目级绑定远端仓库；拉取 `raw/sources/` → 编译 → 提交推送 `raw/` + `wiki/`；APScheduler 每日定时
- **混合搜索** — BM25 + LanceDB 向量 + RRF 融合
- **RAG Chat (SSE)** — 混合搜索 → 图谱 1-hop 扩展 → 流式响应
- **反馈修正** — 对话质量评估 → 编译修复候选 → 人工审核 → 写回 Wiki（含本地 Git 快照回滚）
- **多项目隔离** — 独立 `disk_path` + LanceDB；per-project ingest / git sync 串行锁
- **多用户** — JWT + 项目成员角色；用户 API Token
- **自定义 Agent** — 多项目绑定、工具调用、公开 Chat 端点
- **文档** — 多格式解析；`raw/sources/` 递归列表与内容预览 API

## 快速开始

### 前置条件

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- LLM API Key（OpenAI 兼容 / Ollama 等）
- **Git CLI**（Git 同步与 feedback 写盘快照需要）

### 本地开发（推荐从仓库根目录）

```bash
# 在 llmwiki 根目录
./start-dev.sh
```

Engine 会使用 `data/wiki` 与 `data/engine-db/engine.db`。

### 仅启动 engine（本子目录）

```bash
cd llm-wiki-engine
uv sync
cp .env.example .env   # 可选，根目录 .env 亦可

# 建议指定数据目录（与 monorepo 一致）
export PROJECTS_DIR=../data/wiki
export DATABASE_URL=sqlite+aiosqlite:///$(pwd)/../data/engine-db/engine.db
mkdir -p ../data/wiki ../data/engine-db

uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API 文档：http://localhost:8000/docs  
- 健康检查：http://localhost:8000/health  

### Docker（单服务，开发用）

本子目录含独立 `docker-compose.yml`，仅启动 engine，数据卷为 `./projects`：

```bash
cd llm-wiki-engine
cp .env.example .env
docker compose up -d
```

生产/联调请用**根目录** `docker compose`（含 UI、持久化 `data/`）。

## 配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_API_KEY` | LLM API Key | （空） |
| `EMBEDDING_API_KEY` | Embedding API Key | （空） |
| `JWT_SECRET` | JWT 签名密钥 | `change-me-in-production` |
| `ADMIN_PASSWORD` | 管理员密码 | `admin` |
| `PROJECTS_DIR` | 项目磁盘根目录 | `./projects`（config 默认） |
| `DATABASE_URL` | SQLite 连接串 | `sqlite+aiosqlite:///./data/engine.db` |
| `CONFIG_PATH` | `config.yaml` 路径 | 可选 |

`config.yaml` 示例：

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_base: null

embedding:
  enabled: true
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
```

Ollama 本地示例见原 `config.yaml` 注释；Admin API 可将部分配置写入 DB 覆盖 YAML。

## API 概览

### 认证

```
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
GET  /api/auth/api-token
POST /api/auth/api-token/regenerate
```

### 项目

```
POST   /api/projects
GET    /api/projects
GET    /api/projects/{id}
PATCH  /api/projects/{id}          # 含 Git 同步字段、案例库绑定、反馈开关
DELETE /api/projects/{id}
POST   /api/projects/{id}/members
GET    /api/projects/{id}/members
```

### Git 同步（项目级）

```
POST /api/projects/{id}/git/test     # 测试仓库连接（owner）
POST /api/projects/{id}/git/sync     # 立即同步（成员）
GET  /api/projects/{id}/git/status   # 最近同步状态
```

`PATCH` 项目时可设置：`git_repo_url`、`git_branch`、`git_username`、`git_auth_token`（只写）、`clear_git_auth_token`、`git_sync_enabled`、`git_sync_time` 等。响应含 `git_auth_configured`，不返回 token 明文。

### 文档

```
POST /api/projects/{id}/documents/upload
GET  /api/projects/{id}/documents              # 递归列出 raw/sources/
GET  /api/projects/{id}/documents/content/{path} # 预览正文（PlainText）
```

### 编译（Ingest）

```
POST   /api/projects/{id}/ingest
POST   /api/projects/{id}/ingest/{job_id}/retry
GET    /api/projects/{id}/ingest/status
GET    /api/projects/{id}/ingest/history
DELETE /api/projects/{id}/ingest/{job_id}
```

### Wiki / 搜索 / Chat

```
GET  /api/projects/{id}/wiki
GET  /api/projects/{id}/wiki/overview
GET  /api/projects/{id}/wiki/graph
GET  /api/projects/{id}/wiki/{path}
PUT  /api/projects/{id}/wiki/{path}
POST /api/projects/{id}/search
POST /api/projects/{id}/chat              # SSE
GET  /api/projects/{id}/conversations
```

### 反馈

```
GET  /api/projects/{id}/feedback
POST /api/projects/{id}/feedback/{task_id}/review
POST /api/projects/{id}/feedback/{task_id}/apply
...
```

### Agent / Admin

```
/api/agents/...
/api/public/agents/{id}/chat    # 公开 SSE
/api/admin/...                  # 系统配置（admin）
```

## 项目磁盘布局

每个项目在 `PROJECTS_DIR/<uuid>/`：

```
purpose.md
raw/sources/          # 原始文档（上传 / Git 同步）
wiki/                 # 编译产物
.llm-wiki/            # ingest-cache、LanceDB、checkpoints 等
```

## 测试

```bash
cd llm-wiki-engine
uv sync
uv pip install pytest pytest-asyncio   # 若 venv 未带 dev 依赖
uv run pytest -v
```

## 项目结构

```
llm-wiki-engine/
├── pyproject.toml
├── config.yaml
├── Dockerfile
├── docker-compose.yml      # 仅 engine 单服务
├── .env.example
└── app/
    ├── main.py
    ├── config.py
    ├── database.py
    ├── auth/
    ├── projects/           # CRUD + git_sync.py + 调度注册
    ├── documents/
    ├── ingest/
    ├── embedding/
    ├── search/
    ├── wiki/
    ├── chat/
    ├── agents/
    ├── feedback/
    ├── admin/
    └── llm/
```

## License

MIT

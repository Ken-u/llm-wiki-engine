# LLM Wiki Engine

纯后端 LLM 知识编译引擎 — **Compile-time Knowledge Synthesis**。

不同于传统 RAG（chunk → embedding → 检索原文），本引擎先将原始文档通过 LLM **两步编译** 为结构化 Markdown 知识单元（entity / concept / source summary），再基于高质量知识页面提供检索与问答 API。

## 核心能力

- **两步 CoT Ingest** — Analysis → Generation → FILE 块解析 → LLM 辅助页面合并 → SHA256 增量缓存
- **混合搜索** — BM25 关键词 + LanceDB 向量 + RRF 融合（含文件名 / 标题加分）
- **RAG Chat (SSE)** — 4 阶段检索管线：混合搜索 → 知识图谱 1-hop 扩展 → 上下文预算分配 → 流式响应
- **多项目隔离** — 每项目独立文件系统 + LanceDB 向量库，per-project asyncio.Lock 串行保护 ingest
- **多用户** — JWT 认证 + 项目成员角色（owner / editor / viewer）
- **文档解析** — PDF (PyMuPDF) / DOCX / XLSX / TXT / Markdown / CSV / JSON / YAML
- **LLM 统一接口** — 通过 LiteLLM 支持 OpenAI / Anthropic / Ollama / Azure / 100+ provider

## 快速开始

### 前置条件

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 一个 LLM API Key（OpenAI / 兼容接口 / Ollama 本地）

### 方式一：本地开发（uv）

```bash
# 克隆并进入目录
cd llm-wiki-engine

# 安装依赖
uv sync

# 复制并编辑环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 启动开发服务器
uv run uvicorn app.main:app --reload --port 8000
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 方式二：pip 安装

```bash
cd llm-wiki-engine

# 创建虚拟环境（可选但推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装
pip install -e .

# 安装开发依赖（可选，用于测试和 lint）
pip install -e ".[dev]"

# 复制并编辑环境变量
cp .env.example .env

# 启动
uvicorn app.main:app --reload --port 8000
```

### 方式三：Docker

```bash
cd llm-wiki-engine

# 复制并编辑环境变量
cp .env.example .env

# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f
```

## 配置

所有配置集中在 `config.yaml`，环境变量优先级更高：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_API_KEY` | LLM 服务 API Key | （空） |
| `EMBEDDING_API_KEY` | Embedding 服务 API Key | （空） |
| `JWT_SECRET` | JWT 签名密钥 | `change-me-in-production` |
| `PROJECTS_DIR` | 项目数据存储目录 | `./projects` |

`config.yaml` 支持更细粒度的配置：

```yaml
llm:
  provider: "openai"       # litellm 支持的任何 provider
  model: "gpt-4o-mini"     # 模型名称
  api_base: null            # 自定义端点（Ollama: http://localhost:11434）
  max_context_size: 128000

embedding:
  enabled: true
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
```

**使用 Ollama 本地模型示例：**

```yaml
llm:
  provider: "ollama"
  model: "qwen2.5:14b"
  api_base: "http://localhost:11434"

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  api_base: "http://localhost:11434"
  dimensions: 768
```

## API 概览

### 认证

```
POST /api/auth/register     注册用户
POST /api/auth/login        登录，返回 JWT
GET  /api/auth/me           当前用户信息
```

### 项目管理

```
POST   /api/projects                        创建项目
GET    /api/projects                        我的项目列表
GET    /api/projects/{id}                   项目详情
DELETE /api/projects/{id}                   删除项目
POST   /api/projects/{id}/members           添加成员
GET    /api/projects/{id}/members           成员列表
```

### 文档上传

```
POST /api/projects/{id}/documents/upload    上传文档（multipart, ≤50MB）
GET  /api/projects/{id}/documents           文档列表
```

### 编译（Ingest）

```
POST /api/projects/{id}/ingest              触发编译（单文件或全量）
GET  /api/projects/{id}/ingest/status       进行中的任务
GET  /api/projects/{id}/ingest/history      编译历史
```

### 知识库（Wiki）

```
GET  /api/projects/{id}/wiki                文件树
GET  /api/projects/{id}/wiki/overview       全局概览
GET  /api/projects/{id}/wiki/graph          知识图谱 JSON
GET  /api/projects/{id}/wiki/{path}         读取页面
PUT  /api/projects/{id}/wiki/{path}         编辑页面
```

### 搜索

```
POST /api/projects/{id}/search
Body: { "query": "...", "top_k": 10, "mode": "hybrid|keyword|vector" }
```

### Chat 问答（SSE 流式）

```
POST /api/projects/{id}/chat
Body: { "message": "...", "conversation_id": "..." }
SSE:  data: {"token": "..."}  →  data: {"done": true, "conversation_id": "..."}

GET  /api/projects/{id}/conversations       对话列表
GET  /api/projects/{id}/conversations/{cid} 对话历史
```

## 测试

```bash
# 使用 uv
uv run pytest

# 使用 pip
pytest
```

### 快速冒烟测试

```bash
# 启动服务后：

# 注册
curl -X POST http://localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo","password":"demo123"}'

# 用返回的 token 创建项目
TOKEN="<access_token>"
curl -X POST http://localhost:8000/api/projects \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"My Wiki","slug":"my-wiki","description":"测试项目"}'

# 上传文档
PROJECT_ID="<project_id>"
curl -X POST "http://localhost:8000/api/projects/$PROJECT_ID/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./your-document.pdf"

# 触发编译
curl -X POST "http://localhost:8000/api/projects/$PROJECT_ID/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{}'

# 搜索
curl -X POST "http://localhost:8000/api/projects/$PROJECT_ID/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"你的查询"}'

# Chat（SSE 流式）
curl -N -X POST "http://localhost:8000/api/projects/$PROJECT_ID/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"介绍一下这个知识库的内容"}'
```

## 项目结构

```
llm-wiki-engine/
├── pyproject.toml          # 依赖与构建配置
├── config.yaml             # 运行时配置
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── app/
    ├── main.py             # FastAPI 入口 + 路由注册
    ├── config.py           # YAML + 环境变量配置加载
    ├── database.py         # SQLAlchemy async 数据库
    ├── auth/               # JWT 认证
    ├── projects/           # 项目 CRUD + 成员管理
    ├── documents/          # 文档上传 + 多格式解析
    ├── ingest/             # 两步 CoT 编译管线 + 异步队列
    ├── embedding/          # Markdown 分块 + LanceDB 向量化
    ├── search/             # BM25 + 向量 + RRF 融合
    ├── wiki/               # Wiki CRUD + 知识图谱
    ├── chat/               # RAG Chat (SSE) + 上下文预算
    └── llm/                # LiteLLM 统一封装
```

## License

MIT

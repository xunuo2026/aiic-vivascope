# 问脉 VivaScope

问脉 VivaScope 是一个面向理工科本科生的 AI 口试训练 Web App，用来训练保研复试、科研项目答辩、课程项目展示、实验室面试和技术类实习面试中的连续追问能力。

产品重点不是普通聊天，也不是题库堆砌，而是围绕用户自己的项目经历生成“项目脉络图”和“追问风险雷达”，再进行 6 轮一问一答模拟，每轮给出即时反馈，最后生成结构化复盘报告。

## 核心流程

1. 手动填写面试场景、专业背景、项目经历、训练方向和追问风格，或上传简历 PDF 生成表单建议。
2. 选择训练模式：只练基础知识、只练项目追问、综合模拟。
3. 系统生成项目脉络图、追问风险雷达和必要的知识点清单。
4. 进入 6 轮连续追问，每轮问题都基于项目脉络、风险维度、知识点和上一轮回答。
5. 每轮回答后获得亮点、漏洞、建议补充点和更稳妥的回答框架。
6. 结束后生成总评分、项目易被问穿点、知识薄弱点、表达问题和下一轮训练任务。

## 技术栈

- 后端：FastAPI + Pydantic
- 前端：原生 HTML/CSS/JavaScript
- 模型：阿里云百炼 / DashScope 千问模型，使用 OpenAI-compatible Chat Completions endpoint
- 简历导入：PDF 文本提取 + 千问结构化建议 + 本地启发式兜底
- 会话：后端内存状态 + 前端 localStorage 快照恢复
- 数据库：无
- 登录系统：无

## 环境变量

复制 `.env.example` 为 `.env`，并填入你自己的百炼 API Key。

```bash
cp .env.example .env
```

`.env` 中的关键配置：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
QWEN_MODEL=qwen3.6-max-preview
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
APP_HOST=0.0.0.0
APP_PORT=8000
VIVASCOPE_ALLOW_MOCK=true
```

`QWEN_MODEL` 必须从 `.env` 读取，方便按百炼账号可用模型切换。样例使用 `qwen3.6-max-preview`，实际部署时以你的百炼控制台可调用模型为准。

如果没有配置 `DASHSCOPE_API_KEY` 或 `QWEN_MODEL`，应用会进入本地演示模式，方便录制 UI 和验证流程；正式演示建议配置真实千问模型。

## 本地运行

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问：

```text
http://127.0.0.1:8000
```

快速检查核心闭环：

```bash
python scripts/smoke_test.py
```

## Ubuntu 22.04 部署

目标服务器：

```text
http://182.92.240.206:8000
```

推荐部署目录示例：

```bash
sudo mkdir -p /opt/aiic-vivascope
sudo chown -R $USER:$USER /opt/aiic-vivascope
cd /opt/aiic-vivascope
```

拉取或上传代码后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

如果服务器开启防火墙，需要放行 8000 端口：

```bash
sudo ufw allow 8000/tcp
```

可选 systemd 服务文件见 [deploy/vivascope.service.example](deploy/vivascope.service.example)。

## API 概览

- `GET /health`：健康检查和模型配置状态
- `POST /api/resume/parse`：上传 PDF 简历并生成表单填充建议
- `POST /api/sessions`：创建训练会话并生成初始分析
- `GET /api/sessions/{session_id}`：读取会话
- `POST /api/sessions/{session_id}/answer`：提交一轮回答，获得即时反馈和下一题
- `POST /api/sessions/restore`：从前端快照恢复会话
- `DELETE /api/sessions/{session_id}`：删除会话

## 目录结构

```text
app/
  config.py          环境变量配置
  engine.py          面试分析、追问、反馈和复盘逻辑
  llm_client.py      DashScope 千问调用封装
  main.py            FastAPI 路由和静态文件服务
  schemas.py         请求、响应和会话数据结构
  session_store.py   内存会话存储
  static/            前端页面
deploy/
  vivascope.service.example
```

## 后续扩展建议

- 增加 PDF 简历/项目书解析，但保留文本粘贴作为主流程。
- 增加 Prompt 版本号和训练报告导出。
- 增加更细的专业方向模板，例如 AI、电子、材料、物理实验、自动化控制。
- 增加演示用种子案例，不影响真实训练入口。

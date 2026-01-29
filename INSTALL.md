# 安装指南

## 重要：首次使用需要下载 400MB 模型（必须）

**核心依赖 `rembg` 需要下载 AI 模型文件（约 400MB），首次使用时会自动下载，耗时 2-5 分钟。**

**强烈建议先执行预热命令，提前下载好模型：**

```bash
# 预热命令 - 提前下载 400MB 模型（必须执行）
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

如果不预热，直接在 AI 客户端里使用时会卡在「正在运行工具...」状态 2-5 分钟，体验很差。

---

## 1. 安装 uv（Python 包管理器）

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. 执行预热（下载模型）

```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

等待下载完成（约 400MB）。

---

## 3. 添加 MCP 到你的 AI 客户端

支持 **Claude Code**、**Codex CLI**、**Gemini CLI**、**OpenCode** 等 MCP 客户端。

### Claude Code

```bash
# 添加到当前项目（默认 local scope，仅本项目可用）
claude mcp add --transport stdio docscan -- uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp

# 或添加到用户级别（所有项目可用）
claude mcp add --transport stdio --scope user docscan -- uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp

# 或添加到项目级别（生成 .mcp.json，可提交到仓库团队共享）
claude mcp add --transport stdio --scope project docscan -- uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp
```

验证：
```bash
claude mcp list
# 或在 Claude Code 中运行 /mcp
```

### Codex CLI

编辑 `~/.codex/config.toml`：

```toml
[mcp_servers.docscan]
command = "uvx"
args = ["--from", "git+https://github.com/liusanye/auto_scan", "auto-scan-mcp"]
```

或使用命令：
```bash
codex mcp add docscan
# 然后按提示输入命令: uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp
```

### Gemini CLI

编辑 `~/.gemini/settings.json`：

```json
{
  "mcpServers": {
    "docscan": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/liusanye/auto_scan", "auto-scan-mcp"]
    }
  }
}
```

或使用 FastMCP 快捷安装：
```bash
# 如果你已经安装了 fastmcp
fastmcp install gemini-cli
```

### OpenCode

编辑 `opencode.jsonc`（项目根目录）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "docscan": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/liusanye/auto_scan", "auto-scan-mcp"],
      "enabled": true
    }
  }
}
```

---

## 4. 使用

配置完成后，重启你的 AI 客户端，然后可以直接对话：

- *"帮我扫描 ~/Documents/photo.jpg"*
- *"把 ~/Documents 里的所有文档处理成扫描件"*
- *"扫描这张照片并输出为 PDF"*

---

## 升级方式

```bash
# 清理 uv 缓存获取最新版
uv cache clean
```

然后重启你的 AI 客户端。

---

## 备选：pipx 安装

如果你想永久安装到本地（不依赖 uvx）：

```bash
pipx install git+https://github.com/liusanye/auto_scan

# 预热
auto-scan-mcp --warmup

# 添加到 Claude Code（其他客户端类似）
claude mcp add --transport stdio docscan -- auto-scan-mcp
```

升级：
```bash
pipx upgrade auto-scan
```

---

## 开发安装

如果你想修改代码：

```bash
git clone https://github.com/liusanye/auto_scan.git
cd auto_scan
pip install -e .

# 开发模式运行 MCP
python -m docscan.mcp_server
```


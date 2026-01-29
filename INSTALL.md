# 安装指南

## ⚠️ 重要：首次使用需要下载 400MB 模型

**本工具依赖 `rembg` AI 模型（约 400MB），首次启动时必须下载。**

如果不提前下载，第一次使用会卡住 2-5 分钟，体验很差。

---

## 推荐安装流程（Claude Code 示例）

### 第 1 步：安装 uv（如果还没有）

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 第 2 步：预热（提前下载 400MB 模型）

**这一步会启动 MCP 服务器并下载模型，下载完成后自动退出。**

```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

等待显示 "Warmup complete" 或类似完成提示（约 2-5 分钟，视网速）。

### 第 3 步：配置（告诉 Claude 如何启动 MCP）

**这一步只是写入配置文件，不会下载任何东西。**

```bash
claude mcp add --transport stdio docscan -- uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp
```

### 第 4 步：重启 Claude Code

完全退出 Claude Code，然后重新打开。

### 第 5 步：使用

现在可以直接对话：
- *"帮我扫描 ~/Documents/photo.jpg"*
- *"把 ~/Documents 里的所有文档处理成扫描件"*

---

## 如果不预热会发生什么？

**你会遇到：**

```
你: 帮我扫描这张照片
Claude: [Claude runs a tool...]  ← 卡在这里 2-5 分钟，没有任何提示
```

**原因**：Claude 第一次启动 MCP 服务器时，服务器在后台下载 400MB 模型，Claude 界面只会显示 "running tool"，用户不知道发生了什么。

**所以强烈建议先执行第 2 步预热。**

---

## 其他 AI 客户端配置

### Codex CLI

**预热（和第 1 步一样，只需做一次）：**
```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

**配置：**
编辑 `~/.codex/config.toml`：
```toml
[mcp_servers.docscan]
command = "uvx"
args = ["--from", "git+https://github.com/liusanye/auto_scan", "auto-scan-mcp"]
```

### Gemini CLI

**预热：**
```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

**配置：**
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

### OpenCode

**预热：**
```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

**配置：**
编辑项目根目录的 `opencode.jsonc`：
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

## 升级

```bash
# 清理 uv 缓存，下次使用会拉取最新代码
uv cache clean
```

然后重启你的 AI 客户端。

---

## 备选：pipx 永久安装

如果你不想每次用 uvx，可以永久安装：

```bash
# 安装
pipx install git+https://github.com/liusanye/auto_scan

# 预热（下载模型）
auto-scan-mcp --warmup

# 添加到 Claude Code
claude mcp add --transport stdio docscan -- auto-scan-mcp
```

升级：
```bash
pipx upgrade auto-scan
```

---

## 开发安装

```bash
git clone https://github.com/liusanye/auto_scan.git
cd auto_scan
pip install -e .

# 预热
python -m docscan.mcp_server --warmup

# 开发模式运行
python -m docscan.mcp_server
```

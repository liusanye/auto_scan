# 安装指南

## 重要：首次使用需要下载 400MB 模型（必须）

**核心依赖 `rembg` 需要下载 AI 模型文件（约 400MB），首次使用时会自动下载，耗时 2-5 分钟。**

**强烈建议先执行预热命令，提前下载好模型，避免在 Claude 里卡住：**

```bash
# 预热命令 - 提前下载 400MB 模型（必须执行）
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

如果不预热，直接在 Claude 里使用时会卡在「Claude runs a tool...」状态 2-5 分钟，体验很差。

---

## 推荐方式：uvx（最简单，无需安装）

### 1. 安装 uv（Python 包管理器）

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 执行预热（下载模型）

```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

等待下载完成（约 400MB）。

### 3. 配置 Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：

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

重启 Claude Desktop 即可使用。

---

## 升级方式

### 获取最新版

uvx 默认会缓存，要获取最新代码：

```bash
# 清理缓存
uv cache clean

# 然后重启 Claude Desktop
```

---

## 备选方式：pipx 安装

如果你想永久安装到本地：

```bash
pipx install git+https://github.com/liusanye/auto_scan

# 预热（同样需要）
auto-scan-mcp --warmup
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
```


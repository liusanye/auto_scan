# 安装指南

## 推荐方式：uvx（最简单，无需安装）

### 前置条件

安装 `uv`（Python 包管理器）：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 在 Claude Desktop 中使用

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

**首次使用注意**：首次调用时会自动下载依赖（约 400MB），请耐心等待 2-5 分钟。

### 手动预热（可选）

如果想提前下载好依赖，可以在终端执行：

```bash
uvx --from git+https://github.com/liusanye/auto_scan auto-scan-mcp --warmup
```

## 升级方式

### 自动获取最新版

uvx 默认会缓存，要获取最新代码：

```bash
# 清理缓存
uv cache clean

# 或者使用 @latest 标签
uvx --from git+https://github.com/liusanye/auto_scan@latest auto-scan-mcp
```

### Claude Desktop 升级

1. 清理 uv 缓存：`uv cache clean`
2. 重启 Claude Desktop

## 备选方式：pipx 安装

如果你想永久安装到本地：

```bash
pipx install git+https://github.com/liusanye/auto_scan

# 然后可以直接使用
auto-scan-mcp
```

升级：
```bash
pipx upgrade auto-scan
```

## 开发安装

如果你想修改代码：

```bash
git clone https://github.com/liusanye/auto_scan.git
cd auto_scan
pip install -e .
```

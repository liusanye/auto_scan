# 示例说明

- 仓库不随代码提供可公开样例图片，`source_images/` 仅作为本地回归素材。
- 推荐使用烟囱测试验证裁剪/几何/增强链路：
  ```bash
  python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2 --debug
  ```
- 如需自带样例，请在本目录放入少量图片（单页、双页、旋转、光照不均等），并同步更新 `README.md` / `DEVLOG.md` 记录样本来源与用途。

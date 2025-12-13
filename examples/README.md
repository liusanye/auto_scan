# 示例说明

- 当前项目不随仓库提供真实样例图片，可使用仓库中的 `source_images/` 作为输入示例。
- 推荐通过 `scripts/smoke_test.py` 跑一轮跳过 OCR 的烟囱测试，验证裁剪/几何/增强链路是否正常：
  ```bash
  python scripts/smoke_test.py --input source_images --output outputs_smoke --mode fast --max-files 2
  ```
- 如需自带样例，请在此目录放入少量照片（单页、双页、旋转等），并更新 README/DEVLOG 记录。***

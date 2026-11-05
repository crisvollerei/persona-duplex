# 首页视觉资源

两套封面是本项目自绘的矢量图，随 GitHub 的浅色、深色主题切换。三段节点表示一次前台生成中的职责顺序，虚线表示异步记忆；详细语义以[架构文档](../architecture.md)为准。

- `readme-hero-light.svg`：暖纸色封面。
- `readme-hero-dark.svg`：深色封面。
- `generate_readme_assets.py`：使用 matplotlib 生成可缩放 SVG，文字转为路径，不依赖远端字体或脚本。

在仓库根目录重新生成：

```powershell
python -m pip install -r docs/validation/followup_2026_11_05/requirements.txt
python -B -X utf8 docs/assets/generate_readme_assets.py
```

可加 `--preview-dir data/readme-preview` 输出本地 PNG 预览。图像与生成脚本采用仓库的 MIT 许可。

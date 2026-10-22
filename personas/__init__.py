"""人格配置包。

这些 toml 是**数据**，不是代码 —— 但放进包里才打得进 wheel。
setuptools 的 package-data 只认包内路径，不认 "../personas/*.toml"，
所以把整个目录声明成一个包，配置随包一起分发。

开发时直接读仓库里的文件；装成包之后从 site-packages 里读。
"""
from pathlib import Path

DIR = Path(__file__).resolve().parent


def path_of(name: str) -> Path:
    """按人格名取配置路径（默认补 .toml）。"""
    p = Path(name)
    if p.suffix != ".toml":
        p = p.with_suffix(".toml")
    return DIR / p.name


def available() -> list[str]:
    return sorted(p.stem for p in DIR.glob("*.toml"))

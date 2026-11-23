# 运行与部署

Windows 和 Streamlit Community Cloud 使用同一个入口 `src/eduecon_radar/ui/app.py`。导航、海雾蓝主题、审核表单、论文库、书架和下载由相同的视图与存储服务提供。

| 环境 | 数据与权限 | 生命周期 |
| --- | --- | --- |
| 本地启动脚本 | 从本机配置读取个人库和可选的 API Key，绑定环回地址；支持检索、公开摘要补全与模型分析 | 决定、笔记和任务在本机长期保留 |
| 默认部署 | 从仓库内的公开书目为每个会话建立独立 SQLite 工作区，使用相同的审核、备注、书架与导出；停用外部检索及模型调用 | 会话释放时回收，需长期保存的内容可先导出 |

## Windows

按[本地运行说明](local.md)安装后运行 `.\run_web.ps1`。脚本在当前进程显式选择 `LITERATURE_RADAR_MODE=local`，并把服务绑定到 localhost。直接运行 Streamlit 而不设置模式时，使用临时工作区。

## Streamlit Community Cloud

选定项目仓库和分支，将 **Main file path** 填为 `src/eduecon_radar/ui/app.py`，Python 选择 **3.12**，设置为允许所有人访问。仓库根目录的 `requirements.txt` 安装项目及 Streamlit。**Secrets 留空**。无需上传数据库、个人文件或访问密钥。

默认环境无论是否发现同名环境变量或本机数据库，都只使用 `config/public_papers.json` 构建种子库。服务进程最多同时保留 32 份访客工作区，每条会话备注上限为 1,000 个字符。会话结束或服务重启后，临时资料可被回收。免费托管的首次唤醒时间取决于平台。[平台部署说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 维护

从仓库根目录执行 `python -m pytest` 和 `python -m ruff check src tests`。CI 在 Ubuntu 和 Windows 上从干净检出安装并检查同一入口。共享书目记录 DOI、作者、期刊日期精度、短节选及译文；编者线索独立保存。论文和第三方摘要保留原有权利。个人数据库、缓存、导出、密钥与历史研究记录均保留在 Git 之外。


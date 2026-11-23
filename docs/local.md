# 在本机运行

从一份新克隆开始，使用 Python 3.12。在 Windows PowerShell 的仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m eduecon_radar.cli migrate-database --profile production --database data/eduecon_radar.db --apply
.\run_web.ps1
```

以上迁移命令用于新库。已有研究资料应在应用关闭后先备份数据库、同目录的 `.jobs.sqlite` 与 `.reading.sqlite`，再单独执行升级。启动后从「今日雷达」选择方向，在「关注范围与更新」保存期刊和兴趣词。

检索、人工审核与导出可以直接使用。需要翻译和初筛时，将自己的 DeepSeek 密钥写入本地 `.env`。打开本地服务时任务可续跑，关闭后暂停。

## 接着上次的工作

- 在工作台选期刊与日期后按「获取论文」。可以关闭自动分析，仅查书目。当前批次的模型请求上限为 24 次，含重试。不同模型、输入长度和输出长度对应的金额会有差异。
- 打开「检索记录」找回批次。在「逐篇审核」填写决定和备注，用「保存并下一篇」提交。翻页只切换文章。
- 在论文库选「全部方向」可以复用同一篇文献。方向之间共享书目与翻译，各自保留书架、判断和备注。

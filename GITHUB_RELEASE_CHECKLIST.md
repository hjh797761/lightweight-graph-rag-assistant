# GitHub 发布检查清单

- [ ] `python -m pytest -v` 全部通过。
- [ ] `python -m compileall -q graphrag graphrag_assistant.py dingtalk_server.py scripts tests` 通过。
- [ ] 确定性公开评测能够生成 JSON 和 Markdown。
- [ ] `.env`、API Key、Webhook、个人资料、数据库和模型缓存均未暂存。
- [ ] `git diff --check` 无格式错误。
- [ ] `git status --short --branch` 只包含预期提交。
- [ ] `git fetch origin` 后本地 `main` 与远端没有分叉。
- [ ] 推送前执行 `git pull --ff-only origin main` 并重新运行测试。
- [ ] 使用普通 `git push origin main`，不使用 force push。
- [ ] 推送后用 `git ls-remote origin refs/heads/main` 核对远端 SHA。

如 GitHub secret scanning 或 CI 报告问题，应停止发布并定位原因；不要通过关闭检查或改写远端历史绕过。

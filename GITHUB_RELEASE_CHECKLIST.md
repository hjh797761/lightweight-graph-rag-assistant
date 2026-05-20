# GitHub 发布检查清单

发布前请确认：

- [ ] 已重新生成 Moonshot / Kimi API Key，旧 key 已作废。
- [ ] 已重新生成钉钉机器人 webhook token，旧 token 已作废。
- [ ] 仓库中没有 `.env` 文件。
- [ ] 仓库中没有 `knowledge_base*.json`。
- [ ] 仓库中没有个人 PDF、课程资料、商业报告原文。
- [ ] 仓库中没有 `.deepeval/`、`__pycache__/`、模型缓存、评测大文件。
- [ ] 只发布 `github_release/` 目录中的内容，而不是整个工作区。

推荐发布命令：

```bash
cd github_release
git init
git add .
git commit -m "Initial release of lightweight Graph RAG assistant"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

如果 GitHub 推送时提示 secret scanning，请立即停止推送并重新检查密钥。

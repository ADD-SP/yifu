# yifu

[![CI](https://github.com/ADD-SP/yifu/actions/workflows/ci.yml/badge.svg)](https://github.com/ADD-SP/yifu/actions/workflows/ci.yml)

寻找你在 GitHub 上的义父。

你是否有这样的疑问？你的开源项目突然涨了很多 star，但是不是知道是地里长出来的还是被大佬点名了。

别担心，使用这个工具来分析 star 的来源。

它会把每个 star 分成两类：哪部分是**地里长出来的**，哪部分是**被人安利来的**。

如果真的是被大佬安利来的，难道不应该尊称一声义父么？

<img src="docs/lvbu-1.png" alt="董公待某如此恩重啊" width="320">

## 用法

```bash
uv sync
export GITHUB_TOKEN=ghp_xxx            # 必须：stargazers 名单只对仓库管理员/协作者开放
uv run yifu owner/repo
```

## 原理

根据点了 star 的人和他的 follower 来分析：谁先点的、粉丝有多少，决定后面的 star 是不是他带来的；
前面找不到这样的人，那些 star 就算「地里长出来的」。

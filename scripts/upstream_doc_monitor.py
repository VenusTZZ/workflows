#!/usr/bin/env python3
"""upstream-doc-monitor: 上游文档变化监控检测引擎。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML is required (python -m pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

API_BASE = "https://api.github.com"
USER_AGENT = "upstream-doc-monitor (cosdt-ci-test/workflows upstream doc monitor)"
HTTP_TIMEOUT = 30
RETRY_ATTEMPTS = 3
BACKOFF_SECONDS = (2, 4, 8)
STATE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

# result.json 侧的 open 工单扫描参数；标题前缀与 _sync_events 建票标题一致
TICKET_TITLE_PREFIX = "["
TICKET_LABEL = "upstream-doc-monitor"
TICKET_SCAN_PAGE_SIZE = 100
TICKET_SCAN_MAX_PAGES = 5

# 事件类型 → 工单标题中文标签
_EVENT_LABELS = {
    "change": "文档变化",
    "error": "检测异常",
    "recovery": "异常恢复",
}

# 工单标题正则：{project}] {label} — {target}
_TITLE_RE = re.compile(r"^\[([^\]]+)\] .+ — (.+)$")

GH_USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")

GITHUB_BLOB_URL_RE = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/blob/([^/]+)/(.+)$")


class FatalError(Exception):
    """配置/基线致命错误 → exit 1。"""


class RateLimitError(Exception):
    """GITHUB_TOKEN 配额耗尽 → 本仓库侧运行故障。"""


class TransientError(Exception):
    """网络/5xx 重试耗尽（调用方按采集源归类为 repo_error / fetch_error）。"""


class DocNotFound(Exception):
    """HTTP 404（文档或页面不存在）→ 有效观测，上游侧非致命。"""


class RepoError(Exception):
    """上游仓库级异常（仓库 404/请求持续失败）→ 上游侧非致命（job 绿）。"""


class FetchError(Exception):
    """外部网页不可达（网络失败/5xx/反爬拦截）→ 上游侧非致命（job 绿）。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# HTTP 基础设施
# ---------------------------------------------------------------------------

def _http_once(url: str, *, method: str = "GET", token: str | None = None,
               headers: dict | None = None, data: bytes | None = None):
    """单次 HTTP 请求。返回 (status, headers, body)。网络异常向上抛出。"""
    req_headers = {"User-Agent": USER_AGENT}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, method=method, headers=req_headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 - 读失败不影响分类
            pass
        return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, body


def http_with_retry(url: str, *, method: str = "GET", token: str | None = None,
                    headers: dict | None = None, data: bytes | None = None):
    """带指数退避重试的 HTTP 请求（重试 5xx / 网络失败 / 403 限流）。

    返回首个非重试类结果 (status, headers, body)；
    限流重试耗尽 → RateLimitError；网络/5xx 重试耗尽 → TransientError。
    """
    last_kind, last_detail = None, None
    for attempt in range(RETRY_ATTEMPTS):
        if attempt:
            time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
        try:
            status, resp_headers, body = _http_once(
                url, method=method, token=token, headers=headers, data=data)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_kind, last_detail = "network", repr(exc)
            continue
        if 500 <= status < 600:
            last_kind, last_detail = "http", status
            continue
        if status == 403 and resp_headers.get("x-ratelimit-remaining") == "0":
            last_kind, last_detail = "ratelimit", "X-RateLimit-Remaining=0"
            continue
        return status, resp_headers, body
    if last_kind == "ratelimit":
        raise RateLimitError(f"rate limited after {RETRY_ATTEMPTS} attempts: {url}")
    raise TransientError(f"transient failure after {RETRY_ATTEMPTS} attempts "
                         f"({last_kind}: {last_detail}): {url}")


def gh_headers() -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def gh_get_json(url: str, token: str):
    """GitHub API GET，返回 (status, json_or_None)。网络异常向上抛。"""
    status, _hdrs, body = http_with_retry(url, token=token, headers=gh_headers())
    payload = None
    if body:
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
    return status, payload


# ---------------------------------------------------------------------------
# 配置加载与校验
# ---------------------------------------------------------------------------

def _parse_source_url(where: str, project: str, url: str) -> dict:
    """URL → 内部源结构：github.com blob 链接 → repo_file；其他 http(s) → web_page。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise FatalError(
            f"{where} ({project}): url must be a valid http(s) URL, got {url!r}")
    # 去掉 fragment / query 再匹配 blob 链接：README.md#L1-L5、?plain=1
    # 是浏览器复制的常见形态，不能把 #L1-L5 当文件名拼进 Contents API。
    clean_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", ""))
    match = GITHUB_BLOB_URL_RE.match(clean_url)
    if match:
        repo = f"{match.group(1)}/{match.group(2)}"
        branch = match.group(3)
        spath = match.group(4).strip()
        # key 纳入 repo + branch：同一 project 标签可监控不同仓库/分支的
        # 同名文档，仅用 path 会撞车（FatalError 整轮失败）。
        key = f"{project}::{repo}/{branch}/{spath}"
        return {"key": key, "type": "repo_file", "repo": repo,
                "path": spath, "branch": branch}
    if parsed.netloc.lower() == "github.com":
        raise FatalError(
            f"{where} ({project}): GitHub URLs must point to a single file, "
            "e.g. https://github.com/<owner>/<repo>/blob/<branch>/<path> "
            f"(got {url!r})")
    # 网页类保留 query（部分文档页 query 有语义），只剥 fragment
    web_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    key = f"{project}::{web_url}"
    return {"key": key, "type": "web_page", "url": web_url}


def load_config(path: str) -> tuple[list[dict], str]:
    """解析监控配置并校验；返回 (归一化条目列表, maintainer)。
    致命问题抛 FatalError。"""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FatalError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise FatalError(f"config is not valid YAML: {exc}") from exc

    projects = raw.get("projects")
    if not isinstance(projects, list) or not projects:
        raise FatalError("config must contain a non-empty 'projects' list")

    # maintainer：本 workflow 管理人，条目未填 owner 时的兜底 @ 对象，必填单个用户名
    maintainer = raw.get("maintainer")
    if (not isinstance(maintainer, str)
            or not GH_USERNAME_RE.match(str(maintainer).strip())):
        raise FatalError(
            "config must contain a valid 'maintainer' username "
            f"(got {maintainer!r})")
    maintainer = maintainer.strip()

    entries, seen_keys = [], set()
    for idx, item in enumerate(projects, start=1):
        where = f"projects[{idx}]"
        if not isinstance(item, dict):
            raise FatalError(f"{where}: entry must be a mapping")
        project = item.get("project")
        if not isinstance(project, str) or not project.strip():
            raise FatalError(f"{where}: 'project' label is required")
        project = project.strip()

        owner = item.get("owner")
        if owner is not None:
            owner = str(owner).strip() or None
            if owner and not GH_USERNAME_RE.match(owner):
                raise FatalError(f"{where} ({project}): invalid owner '{owner}'")

        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            if "source" in item:
                raise FatalError(
                    f"{where} ({project}): nested 'source' config is deprecated; "
                    "use a top-level 'url' field instead, e.g. "
                    "url: https://github.com/<owner>/<repo>/blob/<branch>/<path> "
                    "or url: https://example.com/doc")
            raise FatalError(f"{where} ({project}): 'url' is required")
        url = url.strip()

        source = _parse_source_url(where, project, url)
        key = source["key"]
        entry = {"key": key, "project": project, "owner": owner, **source}
        entries.append(entry)

        if key in seen_keys:
            raise FatalError(f"{where} ({project}): duplicate entry key '{key}' "
                             "(same project label + same source)")
        seen_keys.add(key)
    return entries, maintainer


def load_state(path: str) -> dict:
    """读取基线；文件不存在视为首轮。损坏 → FatalError（exit 1）。"""
    state_path = Path(path)
    if not state_path.is_file():
        print(f"state: no baseline at {path} → first run (all first_seen)")
        return {"schema_version": STATE_SCHEMA_VERSION, "entries": {}}
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise FatalError(f"baseline state is corrupted: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), dict):
        raise FatalError("baseline state is corrupted: 'entries' object missing")
    return raw


# ---------------------------------------------------------------------------
# 采集器：按源类型取当前文档哈希 + 错误分类
# ---------------------------------------------------------------------------

def fetch_repo_file_sha(entry: dict, token: str) -> str:
    """repo_file 采集器：Contents API 取 git blob SHA。成功返回 sha；
    文档 404 → DocNotFound；仓库级异常 → RepoError；限流 → RateLimitError。"""
    # 异常文案统一归因：携带定位（repo/path、HTTP 状态）+ url 核对提示，
    # 把「url 哪一段可能错了」直接写进工单，免去人工二次排查。
    hint = "核对 url：仓库/分支/路径任一段有误，或上游已删除/移动该文件"
    target = f"{entry['repo']}/{entry['path']}"
    ref = f"?ref={urllib.parse.quote(entry['branch'])}" if entry.get("branch") else ""
    url = (f"{API_BASE}/repos/{entry['repo']}/contents/"
           f"{urllib.parse.quote(entry['path'])}{ref}")
    try:
        status, _hdrs, body = http_with_retry(url, token=token, headers=gh_headers())
    except RateLimitError:
        raise
    except TransientError as exc:
        raise RepoError(f"repo unreachable ({exc}): {target} — {hint}") from exc
    if status == 200:
        payload = json.loads(body) if body else {}
        sha = payload.get("sha")
        if not sha:
            raise RepoError(f"HTTP {status} from contents API, response missing "
                            f".sha: {target} — {hint}")
        return sha
    if status == 404:
        # 区分文档缺失与仓库级异常：仓库本体可达 → 文档缺失（有效观测）
        try:
            rstatus, _h, _b = http_with_retry(f"{API_BASE}/repos/{entry['repo']}",
                                              token=token, headers=gh_headers())
        except RateLimitError:
            raise
        except TransientError as exc:
            raise RepoError(f"repo probe failed ({exc}): {target} — {hint}") from exc
        if rstatus == 200:
            raise DocNotFound(f"404: {target} — {hint}")
        raise RepoError(f"repo unreachable (HTTP {rstatus}): {target} — {hint}")
    raise RepoError(f"unexpected HTTP {status} from contents API: {target} — {hint}")


def fetch_web_page(entry: dict, prior: dict) -> dict:
    """web_page 采集器：条件请求优先 + SHA-256 兜底。

    返回 {"outcome": "unchanged", "sha": ...}（304 快速判空）或
         {"outcome": "ok", "sha": ..., "etag": ..., "last_modified": ...}。
    404 → DocNotFound；网络/5xx/反爬 → FetchError。"""
    # 异常文案统一归因：与 repo_file 采集器同款约定，附 url 核对提示
    hint = "核对 url 字段拼写；若站点已迁移请更新配置"
    headers = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
    if prior.get("etag"):
        headers["If-None-Match"] = prior["etag"]
    if prior.get("last_modified"):
        headers["If-Modified-Since"] = prior["last_modified"]
    try:
        status, resp_headers, body = http_with_retry(entry["url"], headers=headers)
    except RateLimitError as exc:
        raise FetchError(f"target site rate limited us: {exc} — {hint}") from exc
    except TransientError as exc:
        raise FetchError(f"{exc} — {hint}") from exc
    if status == 304:
        if not prior.get("sha"):
            raise FetchError(
                f"304 but baseline has no sha to reuse: {entry['url']} — {hint}")
        return {"outcome": "unchanged", "sha": prior["sha"]}
    if status == 200:
        digest = hashlib.sha256(body).hexdigest()
        return {"outcome": "ok", "sha": digest,
                "etag": resp_headers.get("etag") or None,
                "last_modified": resp_headers.get("last-modified") or None}
    if status == 404:
        raise DocNotFound(f"404: {entry['url']} — {hint}")
    raise FetchError(f"unexpected HTTP {status} from {entry['url']} — {hint}")


# ---------------------------------------------------------------------------
# 工单同步（交付面）
# ---------------------------------------------------------------------------

class IssueSync:
    """仓库内工单的新建/评论/状态查询。限流时降级为告警+失败返回（不中断）。"""

    def __init__(self, target_repo: str, token: str):
        self.repo = target_repo
        self.token = token

    def get_state(self, issue_api_url: str) -> str | None:
        """返回 issue 状态（open/closed）；不可得（不存在/网络/限流）→ None。"""
        try:
            status, payload = gh_get_json(issue_api_url, self.token)
        except RateLimitError as exc:
            print(f"issue-sync: WARN state check rate limited: {exc}", file=sys.stderr)
            return None
        except TransientError as exc:
            print(f"issue-sync: WARN cannot check issue state: {exc}", file=sys.stderr)
            return None
        if status == 200 and payload:
            return payload.get("state")
        return None

    def create(self, title: str, body: str) -> str | None:
        """新建工单，返回 issue API url；失败返回 None（不中断）。"""
        url = f"{API_BASE}/repos/{self.repo}/issues"
        try:
            status, _h, resp_body = http_with_retry(
                url, method="POST", token=self.token, headers=gh_headers(),
                data=json.dumps({"title": title, "body": body,
                                 "labels": [TICKET_LABEL]}).encode("utf-8"))
        except RateLimitError as exc:
            print(f"issue-sync: WARN create rate limited: {exc}", file=sys.stderr)
            return None
        except TransientError as exc:
            print(f"issue-sync: WARN create failed: {exc}", file=sys.stderr)
            return None
        if status != 201:
            print(f"issue-sync: WARN create returned HTTP {status}", file=sys.stderr)
            return None
        data = json.loads(resp_body) if resp_body else {}
        return data.get("url")

    def comment(self, issue_api_url: str, body: str) -> bool:
        try:
            status, _h, _b = http_with_retry(
                f"{issue_api_url}/comments", method="POST", token=self.token,
                headers=gh_headers(),
                data=json.dumps({"body": body}).encode("utf-8"))
        except RateLimitError as exc:
            print(f"issue-sync: WARN comment rate limited: {exc}", file=sys.stderr)
            return False
        except TransientError as exc:
            print(f"issue-sync: WARN comment failed: {exc}", file=sys.stderr)
            return False
        return status == 201


# ---------------------------------------------------------------------------
# 工单内容
# ---------------------------------------------------------------------------

def _title_target(entry: dict) -> str:
    # repo_file 的标题带 repo/path/branch，与 entry key 对齐，避免
    # 不同仓库同路径的工单标题冲突；列表里可读性更强。
    if entry["type"] == "repo_file":
        return f"{entry['repo']} {entry['path']} ({entry['branch']})"
    return entry["url"]


def _doc_target(entry: dict) -> str:
    return (f"`{entry['repo']}` / `{entry['path']}`"
            if entry["type"] == "repo_file" else entry["url"])


def _doc_links(entry: dict) -> tuple[str, str | None]:
    if entry["type"] == "repo_file":
        branch = entry["branch"]
        doc_url = f"https://github.com/{entry['repo']}/blob/{branch}/{entry['path']}"
        history = f"https://github.com/{entry['repo']}/commits/{branch}/{entry['path']}"
        return doc_url, history
    return entry["url"], None


def _ticket_intro(entry: dict, mention: str,
                  assignee: str | None = None) -> str:
    lines = []
    if mention:
        # @owner 仅出现在正文顶部一次：建票是该轮唯一 GitHub 事件，
        # 处理人只收 1 封邮件（首事件详情就在本正文里）。
        lines += [mention, ""]
    lines += [
        "## 监控项",
        "",
        f"- 项目：`{entry['project']}`",
        f"- 文档：{_doc_target(entry)}",
        f"- 处理人：{assignee if assignee else '（未指定）'}",
        "",
        "> 本 Issue 由 `upstream-doc-monitor` 自动维护：该文档的内容变化与检测"
        "异常都会以评论追加到这里。处理完成后请关闭本 Issue；下次事件会新建 Issue。",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _event_comment(event: str, res: dict, entry: dict, error: dict | None,
                   run_id: str, observed_at: str,
                   mention: str) -> str | None:
    footer = (f"<sub>由 upstream-doc-monitor 自动生成"
              f"（run {run_id}，{observed_at}）</sub>")
    lead = f"{mention}\n\n" if mention else ""
    if event == "change":
        doc_url, history_url = _doc_links(entry)
        lines = [lead + "## 文档变化", "",
                 f"- 观测时间：{observed_at}",
                 f"- 内容哈希：`{res.get('previous_sha', '（无基线）')}` → "
                 f"`{res.get('sha')}`",
                 f"- 文档：{doc_url}"]
        if history_url:
            lines.append(f"- 提交历史：{history_url}")
        lines += ["", "建议动作：核对上游变更 → 更新本仓库对应文档（如受影响）"
                  "→ 关闭本 Issue。", "", footer]
        return "\n".join(lines)
    if event == "error" and error:
        lines = [lead + f"## 检测异常 ({error['error_type']})", "",
                 f"- 观测时间：{observed_at}",
                 f"- 详情：{error.get('message')}"]
        if error.get("doc_url"):
            lines.append(f"- 文档：{error['doc_url']}")
        lines += ["", "建议动作：核查文档路径/URL 是否变更，必要时更新 "
                  "`.github/upstream-doc-monitor.yaml`；处置完成后关闭本 Issue。",
                  "", footer]
        return "\n".join(lines)
    if event == "recovery":
        return (f"{lead}## 异常恢复\n\n此前记录的检测异常已在本轮观测中恢复"
                f"（文档可正常访问）。\n\n{footer}")
    return None


# ---------------------------------------------------------------------------
# 前端产物 result.json：上游版本号 + open 工单关联
# ---------------------------------------------------------------------------

def fetch_upstream_version(repo: str, token: str) -> str | None:
    """三级降级解析上游最新版本号（沿用 quick-start 引擎的 fallback 链约定）。

    1) /releases/latest      —— 多数上游走这条
    2) /releases?per_page=20 —— 覆盖只发 prerelease 的上游（latest 会 404），
                                取第一条非 draft（prerelease 允许）
    3) /tags?per_page=1      —— 覆盖从不发 GitHub Release 的上游

    版本号不是监控核心信号：任何限流/瞬时失败/非预期状态/解析失败都只打
    WARN 并返回 None，绝不上抛，避免影响检测流程与退出码。
    """
    base = f"{API_BASE}/repos/{repo}"
    try:
        # 一级：latest release（404 = 无正式发布，静默降级）
        status, payload = gh_get_json(f"{base}/releases/latest", token)
        if status == 200 and isinstance(payload, dict):
            tag = payload.get("tag_name")
            if tag:
                return str(tag)
        elif status != 404:
            print(f"version: WARN {repo}: /releases/latest returned HTTP "
                  f"{status} → falling back", file=sys.stderr)

        # 二级：最近 20 条 release（draft 不算发布；prerelease 可接受）
        status, payload = gh_get_json(f"{base}/releases?per_page=20", token)
        if status == 200 and isinstance(payload, list) and payload:
            published = [r for r in payload
                         if isinstance(r, dict) and not r.get("draft")]
            pick = published[0] if published else payload[0]
            tag = pick.get("tag_name") if isinstance(pick, dict) else None
            if tag:
                return str(tag)
        elif status != 200:
            print(f"version: WARN {repo}: /releases returned HTTP {status} "
                  "→ falling back", file=sys.stderr)

        # 三级：最新 tag（200 但空列表 = 无 tag，继续到 None）
        status, payload = gh_get_json(f"{base}/tags?per_page=1", token)
        if status == 200 and isinstance(payload, list) and payload:
            first = payload[0]
            tag = first.get("name") if isinstance(first, dict) else None
            if tag:
                return str(tag)
        elif status != 200:
            print(f"version: WARN {repo}: /tags returned HTTP {status} "
                  "→ falling back", file=sys.stderr)
    except RateLimitError as exc:
        print(f"version: WARN {repo}: rate limited ({exc}) → version unknown",
              file=sys.stderr)
        return None
    except TransientError as exc:
        print(f"version: WARN {repo}: transient failure ({exc}) "
              "→ version unknown", file=sys.stderr)
        return None
    print(f"version: WARN {repo}: no version resolved from releases/tags "
          "(or unusable response) → null", file=sys.stderr)
    return None


def fetch_open_ticket_map(repo: str | None, token: str, entries: list[dict],
                          baseline: dict) -> dict[str, str]:
    """扫描本仓库 open 工单并关联到监控项，返回 key → html_url。

    关联优先级：baseline 里记录的 issue_url（API url）精确命中 → 工单标题
    `[upstream-doc-monitor] <project> / <target>` 反解。两条路径都不命中的
    游离工单（条目已从配置删除）直接丢弃，仅打日志供运维排查。
    请求失败时退回到 baseline 的 issue_url 保底显示 pending，不误降级为 unchanged。
    """
    if not repo:
        return {}
    # 关联索引：issue API url → key、(project, target) → key
    by_issue_url, by_title = {}, {}
    for entry in entries:
        key = entry["key"]
        issue_url = baseline.get(key, {}).get("issue_url")
        if issue_url:
            by_issue_url[issue_url] = key
        by_title[(entry["project"], _title_target(entry))] = key

    matched: dict[str, str] = {}
    try:
        for page in range(1, TICKET_SCAN_MAX_PAGES + 1):
            url = (f"{API_BASE}/repos/{repo}/issues?state=open"
                   f"&labels={urllib.parse.quote(TICKET_LABEL)}"
                   f"&per_page={TICKET_SCAN_PAGE_SIZE}&page={page}")
            status, payload = gh_get_json(url, token)
            if status != 200 or not isinstance(payload, list):
                print(f"result: WARN open ticket scan returned HTTP {status} "
                      f"for {repo} (page {page}) → fallback to baseline issue_url",
                      file=sys.stderr)
                break
            for item in payload:
                if not isinstance(item, dict):
                    continue
                # issues API 默认混入 pull request，工单口径只要纯 issue
                if "pull_request" in item:
                    continue
                title = item.get("title") or ""
                m = _TITLE_RE.match(title)
                if not m:
                    continue
                project, target = m.group(1), m.group(2)
                api_url = item.get("url")
                key = by_issue_url.get(api_url) if api_url else None
                if key is None:
                    key = by_title.get((project, target))
                if key is None:
                    print(f"result: orphan open ticket #{item.get('number')} "
                          f"({title}) — no matching monitor entry")
                    continue
                html_url = item.get("html_url") or (
                    _html_url(repo, api_url) if api_url else None)
                if html_url:
                    matched[key] = html_url
            if len(payload) < TICKET_SCAN_PAGE_SIZE:
                break
        else:
            # for 循环正常结束（未被 break）说明打到第 5 页且每页都满：
            # 超过 500 条 open 工单，提示可能截断
            print(f"result: WARN open ticket scan hit page limit "
                  f"({TICKET_SCAN_MAX_PAGES} pages); "
                  "ticket map may be incomplete",
                  file=sys.stderr)
    except RateLimitError as exc:
        print(f"result: WARN open ticket scan rate limited ({exc}) "
              "→ fallback to baseline issue_url", file=sys.stderr)
    except TransientError as exc:
        print(f"result: WARN open ticket scan failed ({exc}) "
              "→ fallback to baseline issue_url", file=sys.stderr)

    # 扫描失败或截断时，baseline 记录的 issue_url 保底写入 pending
    for entry in entries:
        key = entry["key"]
        if key not in matched:
            issue_url = baseline.get(key, {}).get("issue_url")
            if issue_url:
                html_url = _html_url(repo, issue_url)
                if html_url:
                    matched[key] = html_url
    return matched


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True,
                        help="监控配置文件 .github/upstream-doc-monitor.yaml")
    parser.add_argument("--state",
                        help="基线状态文件（actions/cache 持久化）")
    parser.add_argument("--output-dir",
                        help="内部报告 report.json 输出目录")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                        help="Issue 目标仓库 owner/repo（缺省取 GITHUB_REPOSITORY）")
    parser.add_argument("--validate-only", action="store_true",
                        help="只校验配置文件合法性后退出"
                             "（供 PR 触发的配置校验 workflow 使用）")
    args = parser.parse_args(argv)

    # 配置合法性校验：与运行期共用 load_config，规则单一来源
    if args.validate_only:
        try:
            entries, _ = load_config(args.config)
        except FatalError as exc:
            print(f"FATAL(config): {exc}", file=sys.stderr)
            return 1
        print(f"config: OK ({len(entries)} entries)")
        return 0

    if not args.state or not args.output_dir:
        parser.error("--state and --output-dir are required unless "
                     "--validate-only is set")

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GH_TOKEN (or GITHUB_TOKEN) is required", file=sys.stderr)
        return 1

    try:
        entries, maintainer = load_config(args.config)
    except FatalError as exc:
        print(f"FATAL(config): {exc}", file=sys.stderr)
        return 1
    try:
        state = load_state(args.state)
    except FatalError as exc:
        print(f"FATAL(state): {exc}", file=sys.stderr)
        return 1
    baseline = state.setdefault("entries", {})

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    trigger = os.environ.get("GITHUB_EVENT_NAME", "manual")
    observed_at = utc_now()
    print(f"monitor: {len(entries)} entries, trigger={trigger}, run_id={run_id}")

    repo_file_blocked = False   # 限流后剩余 repo_file 项直接跳过（不再打 API）
    rate_limited_seen = False
    results = {}                # key → {"status": ..., "error_type": ..., ...}

    for idx, entry in enumerate(entries, start=1):
        key = entry["key"]
        prior = baseline.get(key, {})
        label = f"[{idx}/{len(entries)}] {key}"
        if entry["type"] == "repo_file" and repo_file_blocked:
            results[key] = {"status": "error", "error_type": "rate_limited",
                            "error": "skipped after quota exhaustion"}
            print(f"{label}: rate_limited (skipped)")
            continue
        try:
            if entry["type"] == "repo_file":
                sha = fetch_repo_file_sha(entry, token)
                web = None
            else:
                web = fetch_web_page(entry, prior)
                sha = web["sha"]
                if web["outcome"] == "unchanged":
                    results[key] = {"status": "unchanged", "sha": sha,
                                    "web": web}
                    print(f"{label}: unchanged (304)")
                    continue
            if prior.get("sha") is None:
                results[key] = {"status": "first_seen", "sha": sha, "web": web}
                print(f"{label}: first_seen sha={sha[:12]}...")
            elif prior["sha"] != sha:
                results[key] = {"status": "changed", "sha": sha,
                                "previous_sha": prior["sha"], "web": web}
                print(f"{label}: changed {prior['sha'][:12]}... → {sha[:12]}...")
            else:
                results[key] = {"status": "unchanged", "sha": sha, "web": web}
                print(f"{label}: unchanged")
        except DocNotFound as exc:
            results[key] = {"status": "error", "error_type": "doc_not_found",
                            "error": str(exc)}
            print(f"{label}: ERROR doc_not_found ({exc})")
        except RateLimitError as exc:
            rate_limited_seen = True
            if entry["type"] == "repo_file":
                repo_file_blocked = True
            results[key] = {"status": "error", "error_type": "rate_limited",
                            "error": str(exc)}
            print(f"{label}: ERROR rate_limited ({exc})")
        except (RepoError, FetchError) as exc:
            etype = "repo_error" if isinstance(exc, RepoError) else "fetch_error"
            results[key] = {"status": "error", "error_type": etype,
                            "error": str(exc)}
            print(f"{label}: ERROR {etype} ({exc})")

    # "完成观测" 口径：拿到确定结果即算（含 404 这种有效观测）；
    # repo_error / fetch_error / rate_limited = 未能观测。
    observed = sum(
        1 for r in results.values()
        if r["status"] in ("changed", "unchanged", "first_seen")
        or r.get("error_type") == "doc_not_found")

    # ---- 工单同步 ----
    syncer = None
    if args.repo:
        syncer = IssueSync(args.repo, token)
    else:
        print("issue-sync: no target repo (set --repo or GITHUB_REPOSITORY); "
              "ticket sync skipped")

    changes, errors = [], []
    tickets = {"created": 0, "commented": 0}
    for entry in entries:
        key = entry["key"]
        res = results.get(key, {})
        prior = baseline.get(key, {})
        status = res.get("status")

        if status in ("changed", "unchanged", "first_seen"):
            doc_url, history_url = _doc_links(entry)
            change = {"project": entry["project"], "owner": entry["owner"],
                      "source": entry["type"], "status": status,
                      "current_sha": res.get("sha"), "doc_url": doc_url}
            if entry["type"] == "repo_file":
                change.update({"repo": entry["repo"], "doc_path": entry["path"],
                               "history_url": history_url})
            if status == "changed":
                change["previous_sha"] = res.get("previous_sha")

            # 事件序列：先异常恢复、后文档变化（spec 固定顺序）
            events = []
            if prior.get("last_event") == "error":
                events.append("recovery")
            if status == "changed":
                events.append("change")
            change["events"] = list(events)

            new_entry = dict(prior)
            new_entry.update({"sha": res["sha"], "date": observed_at,
                              "last_event": "ok"})
            new_entry.pop("last_error_type", None)
            web = res.get("web")
            if web:
                # 仅在服务器给出新验证器时覆盖（304 时保留旧值）
                if web.get("etag"):
                    new_entry["etag"] = web["etag"]
                if web.get("last_modified"):
                    new_entry["last_modified"] = web["last_modified"]

            if events and syncer:
                issue_url, action, synced = _sync_events(
                    syncer, entry, prior, events, res, run_id, observed_at,
                    tickets, maintainer=maintainer)
                change["ticket_action"] = action
                if synced:
                    if issue_url:
                        change["ticket_url"] = _html_url(args.repo, issue_url)
                        new_entry["issue_url"] = issue_url
                else:
                    # 同步失败（create/comment 未送达）：保留旧基线，
                    # 下一轮重新检测并重新同步，不吞变化。
                    print(f"result: WARN event sync failed for {key}, "
                          "baseline kept for retry", file=sys.stderr)
                    results[key]["sync_failed"] = True
                    baseline[key] = prior
                    changes.append(change)
                    continue
            elif events:
                change["ticket_action"] = "skipped(no-repo)"

            baseline[key] = new_entry
            changes.append(change)
            continue

        if status == "error":
            etype = res["error_type"]
            err = {"project": entry["project"], "owner": entry["owner"],
                   "source": entry["type"], "error_type": etype,
                   "message": res.get("error")}
            if entry["type"] == "repo_file":
                err.update({"repo": entry["repo"], "doc_path": entry["path"],
                            "doc_url": f"https://github.com/{entry['repo']}"
                                       f"/blob/{entry['branch']}/{entry['path']}"})

            new_entry = dict(prior)  # 保留旧哈希（文档回来时可比对）

            if etype == "rate_limited":
                # 本仓库侧凭证问题：不发通知（job 红即告警），基线不动
                errors.append(err)
                baseline[key] = new_entry
                continue

            same_error = (prior.get("last_event") == "error"
                          and prior.get("last_error_type") == etype)
            if not same_error and syncer:
                issue_url, action, synced = _sync_events(
                    syncer, entry, prior, ["error"], res, run_id, observed_at,
                    tickets, error=err, maintainer=maintainer)
                err["ticket_action"] = action
                if synced:
                    if issue_url:
                        err["ticket_url"] = _html_url(args.repo, issue_url)
                        new_entry["issue_url"] = issue_url
                else:
                    # 同步失败：保留旧基线（last_event/last_error_type 不更新），
                    # 下一轮重新检测并重新同步，避免异常事件被静默吞掉。
                    print(f"result: WARN error event sync failed for {key}, "
                          "baseline kept for retry", file=sys.stderr)
                    baseline[key] = prior
                    errors.append(err)
                    continue

            new_entry.update({"date": observed_at, "last_event": "error",
                              "last_error_type": etype})
            baseline[key] = new_entry
            errors.append(err)
        # rate_limited-skipped 条目：基线不动（results 中已计）

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": observed_at,
        "run_id": run_id,
        "trigger": trigger,
        "summary": {
            "monitor_entries": len(entries),
            "by_source": {
                "repo_file": sum(1 for e in entries if e["type"] == "repo_file"),
                "web_page": sum(1 for e in entries if e["type"] == "web_page"),
            },
            "changed": sum(1 for c in changes if c["status"] == "changed"),
            "first_seen": sum(1 for c in changes if c["status"] == "first_seen"),
            "unchanged": sum(1 for r in results.values()
                             if r["status"] == "unchanged"),
            "errors": len(errors),
            "tickets": tickets,
        },
        "changes": changes,
        "errors": errors,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"report: written {report_path}")
    print(f"summary: {json.dumps(report['summary'], ensure_ascii=False)}")

    # ---- 前端产物 result.json（裸数组，每项恰 5 字段）----
    # 与 report.json 同目录的第二份产物：report.json 保持内部详报口径不变，
    # result.json 只给前端渲染用的极简视图。工单映射在工单同步之后取，
    # 本轮新建的工单即计入 pending；所有采集失败均已在上游函数内降级。
    open_tickets = fetch_open_ticket_map(args.repo, token, entries, baseline)
    version_cache: dict[str, str | None] = {}   # 同 repo 多条目只查一次
    result_items = []
    for entry in entries:
        key = entry["key"]
        if entry["type"] == "repo_file":
            repo = entry["repo"]
            if repo not in version_cache:
                version_cache[repo] = fetch_upstream_version(repo, token)
            version, doc = version_cache[repo], _doc_links(entry)[0]
        else:
            version, doc = None, entry["url"]   # 网页类无版本号，不打 API
        ticket = open_tickets.get(key)
        # 状态优先级：本轮检测失败 > 存在未关闭工单 > 其余一律 unchanged
        if (results.get(key, {}).get("status") == "error"
                or results.get(key, {}).get("sync_failed")):
            item_status = "error"
        elif ticket:
            item_status = "pending"
        else:
            item_status = "unchanged"
        result_items.append({"project": entry["project"], "doc": doc,
                             "version": version, "status": item_status,
                             "ticket": ticket})
    result_path = out_dir / "result.json"
    result_path.write_text(
        json.dumps(result_items, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"result: written {result_path}")

    state_path = Path(args.state)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"state: written {state_path}")

    # ---- 退出码（按故障域） ----
    if len(entries) > 0 and observed == 0:
        print("exit 2: no entry completed observation (environment failure?)")
        return 2
    if rate_limited_seen:
        print("exit 2: rate limited — remaining GitHub API detections skipped")
        return 2
    return 0


def _html_url(target_repo: str, issue_api_url: str) -> str:
    """API url → 浏览器 url（issues/<n>）。"""
    tail = issue_api_url.rstrip("/").rsplit("/", 1)[-1]
    return f"https://github.com/{target_repo}/issues/{tail}"


def _sync_events(syncer: IssueSync, entry: dict, prior: dict, events: list[str],
                 res: dict, run_id: str, observed_at: str, tickets: dict,
                 error: dict | None = None,
                 maintainer: str | None = None) -> tuple[str | None, str, bool]:
    """为一个监控条目同步工单；返回 (issue_url, action, synced)。

    action ∈ created（本轮新建）/ commented（在 open 工单追加）/ none（无工单或失败）。
    synced=False 表示本次事件实际未送达（create/comment 失败）：调用方必须
    保留旧基线，让下一轮重新检测并重新同步，避免静默吞掉变化。
    """
    # 首事件类型作为标题标签：error 事件用具体错误类型，其他取第一个非 recovery 事件
    if events == ["error"] and error and error.get("error_type"):
        primary_label = _EVENT_LABELS.get(error["error_type"], "检测异常")
    else:
        primary = next((e for e in events if e != "recovery"), events[0] if events else "error")
        primary_label = _EVENT_LABELS.get(primary, primary)
    title = f"[{entry['project']}] {primary_label} — {_title_target(entry)}"
    # 条目未填 owner 时兜底 @ maintainer（本 workflow 管理人）
    assignee = entry["owner"] or maintainer
    mention = f"@{assignee}" if assignee else ""
    issue_url = prior.get("issue_url")

    if issue_url:
        state = syncer.get_state(issue_url)
        if state != "open":
            issue_url = None  # 已关闭或已删除：本轮事件新建

    created = False
    if not issue_url:
        # 仅"异常恢复"单事件且无 open 工单 → 不为恢复单独建票（噪音）；
        # 状态由调用方置 ok 即可。
        if events == ["recovery"]:
            return None, "none", True
        # 首事件（本轮全部事件）并入正文：建票成为唯一 GitHub 事件，
        # 处理人只收 1 封邮件；后续轮次的事件才走评论追加。
        sections = []
        for event in events:
            section = _event_comment(event, res, entry, error, run_id,
                                     observed_at, "")
            if section:
                sections.append(section)
        body = _ticket_intro(entry, mention, assignee)
        if sections:
            body = body.rstrip("\n") + "\n\n" + "\n\n".join(sections) + "\n"
        issue_url = syncer.create(title, body)
        if not issue_url:
            print(f"ticket: WARN create failed for {entry['key']}", file=sys.stderr)
            return None, "none", False
        created = True
        print(f"ticket: created {title}")
    else:
        ok = True
        for event in events:
            body = _event_comment(event, res, entry, error, run_id,
                                  observed_at, mention)
            if body and syncer.comment(issue_url, body):
                print(f"ticket: commented '{event}' → {issue_url}")
            elif body:
                print(f"ticket: WARN comment failed '{event}' → {issue_url}",
                      file=sys.stderr)
                ok = False
        if not ok:
            return issue_url, "none", False
    action = "created" if created else "commented"
    tickets[action] = tickets.get(action, 0) + 1
    return issue_url, action, True


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())

"""Quick-start-Ascend 文档测试：基于 ``MarkdownDocTestBase`` 契约的端到端用例。

被测文档：``projects/areno/docs/Quick-start-Ascend.md``
（遵循 ``docs/markdown_doc_test_label.md`` 契约：每个 ``shell`` / ``python`` 代码块
带 ``#test`` / ``#test-setup`` / ``#test-result`` 标签之一，以及 ``id=`` / ``store=``
/ ``load='x>>y'`` / ``fuzzy='xxx'`` 参数）。

运行：``python -m unittest tests.test_quick_start_ascend -v 2>&1``

环境变量（由 quick-start 引擎 workflow ``quick-start-template.yml`` 注入，
``areno-quick-start.yml`` 触发）：
    ``MONITORED_DOC_URL``         必需；被测文档的 URL。
    ``UPSTREAM_REF``              必需；最新 release tag。它的值由
                                  ``#test-setup store="upstream_ref"`` 块的 stdout
                                  捕获进 ``captures``，再替换到文档里 ``<ref>`` 处。
    ``NPU_READY=true``            必需，否则整类跳过。端到端只在 NPU runner 上跑：
                                  本地/普通 ubuntu runner 没有 ``/dev/davinci*``，
                                  硬跑会在 ``import torch_npu`` 处失败。
"""

from __future__ import annotations

import os
import subprocess
import unittest

from workflows.markdown_doc_test_base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'``（忽略大小写）-> True；其余（含未设）-> False。"""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """``NPU_READY=true`` 时返回 True，解除跳过。"""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` 端到端测试：取文档 -> 校验契约 -> 按序执行
    ``#test-setup`` / ``#test`` -> 与 ``#test-result`` 比对。

    范围：NPU 栈检查（torch / torch_npu / 设备数）+ 从引擎解析到的最新 release
    源码安装（安装器检测到 ``torch_npu`` 后走 ``requirements/npu.txt`` 并编译
    ``areno.accel._areno_accel_npu``，无回退，故构建必须在 CANN 9.0.0 镜像上成功）
    + CLI 入口冒烟 + GSPO-on-GSM8K 1 步 RLVR 冒烟（Qwen3-0.6B，ModelScope：rollout
    + reward + train + 存 checkpoint）+ 对该 checkpoint 的 OpenAI 兼容 serve
    （后台启动、/health 轮询、/v1/models 与 /v1/chat/completions 检查、teardown）。
    """

    # 60 分钟：``uv pip install -e . --no-build-isolation`` 里的 AscendC 内核 +
    # torch_npu 扩展构建占大头；之后 1 步 GSPO 冒烟（0.6B，8 × 256 token rollout）
    # 在冷下载模型/数据后再加几分钟。
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'  # 被看护源是 cosdt-ci-test 下的 fork
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # 通用 [ERROR] + Traceback
        'applicaiton exception',  # CANN 驱动的拼写（sic）
        'ERR99999',  # CANN 不可恢复错误哨兵
    )

    # CANN toolkit：source 一次拿到 ASCEND_HOME / LD_LIBRARY_PATH 等。路径写死，
    # 与 ``areno-quick-start.yml`` 的 ``image:`` 输入绑定的镜像一致。
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment：CANN 环境 + uv（torch / torch_npu / AReno 的安装由
    # 文档的 `#test-setup` / `#test` 块负责）
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """source CANN 环境并安装 uv。

        ``torch / torch_npu`` 与 ``areno`` 由文档的 ``#test-setup`` / ``#test`` 块
        负责；这里只做必须在那些块之前完成的 runner 侧脚手架。

        类级准备，由 ``setUpClass`` 触发一次（不是每个测试方法都跑的
        ``unittest.TestCase.setUp``）。
        """
        # 0) CANN 环境：source set_env.sh 并合并进 os.environ。workflow 级的
        # jobs.env / steps.env 优先于 CANN 默认值。文档每个块子进程都继承该环境，
        # 于是 ``uv pip install -e .`` 里的 AscendC 构建即使早于文档自己的 source
        # 行也能看到 ASCEND_HOME_PATH。
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) uv：文档的安装块用 ``uv pip install -e .``，它处理 PEP 517 构建依赖
        # （以及 flash-linear-attention 的 git 直接引用）比 pip 更稳。继承 yml
        # job 级 env 的 ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``（集群缓存 + trusted
        # host）。
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

    # ----------------------------------------------------------
    # 测试入口
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """每类跑一次环境准备：CANN 环境 + uv。

        ``areno`` 不在这里装——原因见 ``prepare_environment``。

        ``@unittest.skipIf`` 只跳过测试*方法*，``setUpClass`` 本身总会执行；下面
        ``if _e2e_enabled()`` 才是真正防止 ``NPU_READY`` 未设时触发重准备的部分。
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """模板方法入口。基类 ``run_template()`` 跑完整的 ``pre_process`` ->
        ``parse`` -> ``execute`` -> ``post_process``；``prepare_environment`` 由
        ``setUpClass`` 触发一次，不在 ``run_template`` 里。"""

        self.run_template()


if __name__ == '__main__':
    unittest.main()

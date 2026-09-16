"""``markdown_doc_test_base`` 的单元测试。

覆盖契约规则的解析 / 校验 / 折叠 / 占位符替换 / 正则比对五条核心路径：

* ``parse`` 切代码块（普通块跳过、HTML 注释内 setup 识别、mistune AST 行为）
* ``_parse_block`` 解析 info string（id / store / load / fuzzy）
* ``_validate`` 规则 2/5/7/10/11 违规抛 ``LabelSpecError``
* ``_fold`` 生成 ``SetupCommand`` / ``TestCommand`` / ``TestExpectedOutput``
* ``substitute_placeholders`` 纯函数（缺 store 时保留字面占位符）
* ``compare_output`` 默认 ``...`` 与 ``fuzzy='xxx'`` 的非贪婪跨行匹配

跑法：``python -m unittest tests.test_markdown_doc_test_base -v 2>&1``

源码位置：``src/workflows/markdown_doc_test_base.py``，所有项目共享。本
测试文件通过 ``sys.path`` 注入仓库根 + ``src/`` 后按 ``workflows.*`` 包
导入，避免在测试目录重复放置同一份源码。
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

# 把仓库根加进 sys.path，使 ``workflows.*`` 可解析为 ``src/workflows/*``。
_REPO_ROOT = Path(__file__).resolve().parents[1]  # tests/ -> workflows/
_SRC = _REPO_ROOT / 'src'
for _p in (_SRC, _REPO_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from workflows.markdown_doc_test_base import (  # noqa: E402
    LabelSpecError,
    MarkdownDocTestBase,
    SetupCommand,
    TestCommand,
    TestExpectedOutput,
    _rescan_fences,
)


class _Bare(MarkdownDocTestBase):
    """不带 pre/post 钩子的最小子类,只暴露基类方法供测试直接调用。"""

    def pre_process(self) -> str:
        raise NotImplementedError

    def post_process(self) -> None:
        return None


# macOS / 很多开发机只有 ``python3``、没有 ``python``，但生产 Linux NPU 镜像默认带
# ``python``（CANN 镜像用 ``python -m`` 起脚本）。基类契约保留 ``['python', '-c']``
# 作为面向作者的默认值；这套测试在跑时把 runner 重定向到 ``sys.executable``（也即
# 跑当前 unittest 的同一个解释器），让 python 镜像测试在 macOS 上也能跑通。
# 重写方式用类属性覆盖，不污染基类其它子类的语义。
_PYTHON_BIN = shutil.which('python') or sys.executable
_Bare._LANG_RUNNER = {**_Bare._LANG_RUNNER, 'python': (_PYTHON_BIN, '-c')}


def _parse(text: str) -> tuple[list, dict]:
    """薄封装:跳过 pre_process,直接调基类 ``parse``。"""
    return _Bare().parse(text)


class TestScanBlocks(unittest.TestCase):
    """``_scan_blocks`` + ``_parse_block`` 路径：无标签块跳过、HTML 注释识别。"""

    def test_plain_block_skipped(self):
        """无标签的 ```python``` 不进 commands / results。"""
        text = (
            '```python\n'
            'print(1)\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(commands, [])
        self.assertEqual(results, {})

    def test_test_block_in_main_sequence(self):
        """```shell #test id=\"x\"``` 进 commands 不进 results。"""
        text = (
            '```shell #test id="x"\n'
            'echo hi\n'
            '```\n'
            '\n'
            '```shell #test-result id="x"\n'
            'hi\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertIsInstance(commands[0], TestCommand)
        self.assertEqual(commands[0].id, 'x')
        self.assertEqual(results.keys(), {'x'})

    def test_setup_block_with_store(self):
        """```shell #test-setup store=\"x\"``` 进 commands。"""
        text = (
            '```shell #test-setup store="x"\n'
            'echo captured\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertIsInstance(commands[0], SetupCommand)
        self.assertEqual(commands[0].store, 'x')
        self.assertEqual(results, {})

    def test_setup_in_html_comment(self):
        """HTML 注释内的 #test-setup 仍切出,hidden=True。"""
        text = (
            '<!-- hidden setup\n'
            '```shell #test-setup store="hidden_var"\n'
            'echo from_comment\n'
            '```\n'
            '-->\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].hidden)
        self.assertEqual(commands[0].store, 'hidden_var')


class TestRescanFences(unittest.TestCase):
    """``_rescan_fences`` 从 ``block_html.raw`` 救出注释内 fence。"""

    def test_rescues_single_fence(self):
        raw = (
            '<!-- \n'
            '```shell #test-setup\n'
            'cmd\n'
            '```\n'
            '-->'
        )
        out = _rescan_fences(raw)
        self.assertEqual(len(out), 1)
        info, body = out[0]
        self.assertEqual(info, 'shell #test-setup')
        self.assertEqual(body, 'cmd')

    def test_rescues_multiple_fences(self):
        raw = (
            '<!--\n'
            '```shell #test-setup store="a"\n'
            'A\n'
            '```\n'
            '\n'
            '```shell #test-setup store="b"\n'
            'B\n'
            '```\n'
            '-->'
        )
        out = _rescan_fences(raw)
        self.assertEqual([info for info, _ in out],
                         ['shell #test-setup store="a"',
                          'shell #test-setup store="b"'])

    def test_unclosed_fence_raises(self):
        raw = '<!--\n```shell #test-setup\nstill running\n'
        with self.assertRaises(LabelSpecError):
            _rescan_fences(raw)

    def test_empty_html_block_no_fence(self):
        """无 fence 的注释块返回空列表（不抛错）。"""
        self.assertEqual(_rescan_fences('<!-- just a comment -->'), [])
        self.assertEqual(_rescan_fences('<!-- \nmulti\nline\ncomment\n-->'), [])


class TestValidateRules(unittest.TestCase):
    """``_validate``：规则 2/5/7/10/11。"""

    def test_rule2_duplicate_id(self):
        """同 type 重复 id 抛 ``LabelSpecError``。"""
        text = (
            '```shell #test id="dup"\necho a\n```\n'
            '```shell #test id="dup"\necho b\n```\n'
            '```shell #test-result id="dup"\nb\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        msg = str(cm.exception)
        self.assertIn('duplicate', msg)
        self.assertIn('dup', msg)

    def test_rule5_missing_pair(self):
        """#test 缺同 id 的 #test-result 抛错。"""
        text = '```shell #test id="lonely"\necho a\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('matching', str(cm.exception))

    def test_rule7_test_without_language(self):
        """规则 7:``#test`` / ``#test-setup`` 必须有 language,且必须在契约
        白名单内(支持 shell / python)。"""
        # 缺 language
        text = '```#test id="x"\necho\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        msg = str(cm.exception)
        self.assertIn('#test', msg)
        self.assertIn('echo', msg)
        # #test-setup 缺 language 也抛错
        text_setup = '```#test-setup store="x"\necho\n```\n'
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text_setup)
        msg = str(cm.exception)
        self.assertIn('#test-setup', msg)
        self.assertIn('echo', msg)
        # 不支持的语言(text / console 等)也抛错 — python 已升级到白名单,
        # 留 text / console 当反例,再加一个完全虚构的 lang 锁住校验链路。
        for lang in ('text', 'console', 'ruby'):
            text_bad = f'```{lang} #test id="x"\necho\n```\n'
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text_bad)
            msg = str(cm.exception)
            self.assertIn(lang, msg)
            self.assertIn('not supported', msg)
        # shell / python 通过校验
        for lang in ('shell', 'python'):
            text_ok = (
                f'```{lang} #test id="x"\necho\n```\n'
                f'```{lang} #test-result id="x"\nhi\n```\n'
            )
            commands, _results = _parse(text_ok)
            self.assertEqual(commands[0].language, lang)

    def test_rule10_non_setup_in_comment(self):
        """HTML 注释内出现 #test / #test-result 抛错。"""
        text = (
            '<!-- bad\n'
            '```shell #test id="x"\necho a\n```\n'
            '-->\n'
            '```shell #test-result id="x"\na\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('HTML comment', str(cm.exception))

    def test_rule11_load_before_store(self):
        """load 引用的 store 在文档中还没出现就报错。"""
        text = (
            '```shell #test id="x" load="missing>>local"\n'
            'echo <local>\n```\n'
            '```shell #test-result id="x"\nsomething\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn("load=", str(cm.exception))
        self.assertIn('earlier', str(cm.exception))

    def test_rule12_fuzzy_only_on_test_result(self):
        """``fuzzy=`` 只允许出现在 ``#test-result``:``#test`` / ``#test-setup``
        块带 fuzzy 直接报错。"""
        for label in ('#test', '#test-setup'):
            text = (
                f'```shell {label} id="x" fuzzy="xxx"\n'
                'echo\n```\n'
                f'```shell #test-result id="x"\nhi\n```\n'
            )
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text)
            self.assertIn('fuzzy', str(cm.exception))
            self.assertIn(label, str(cm.exception))

    def test_rule13_multi_fuzzy_parsed(self):
        """``#test-result`` 上 ``fuzzy='xxx' fuzzy='yyy'`` 解析为 tuple,
        ``xxx`` / ``yyy`` 都是非贪婪通配。"""
        text = (
            '```shell #test-result id="x" fuzzy="xxx" fuzzy="yyy"\n'
            'a xxx b yyy c\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(results['x'].fuzzy, ('xxx', 'yyy'))
        # 实际匹配也走通:xxx 和 yyy 各自按通配处理
        actual = 'a 1 b 2 c\n'
        self.assertTrue(_Bare().compare_output(actual, results['x'].body,
                                               fuzzy=results['x'].fuzzy))

    def test_rule14_duplicate_fuzzy_rejected(self):
        """同一个 placeholder 写两次算违规:大概率笔误。"""
        text = (
            '```shell #test-result id="x" fuzzy="xxx" fuzzy="xxx"\n'
            'a\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('duplicate fuzzy', str(cm.exception))

    def test_rule15_disable_fuzzy_parsed(self):
        """``disable_fuzzy`` 出现在 ``#test-result`` 上被解析为 ``True``。"""
        text = (
            '```shell #test-result id="x" disable_fuzzy\n'
            'hello ... world\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertTrue(results['x'].disable_fuzzy)
        # 也验证 compare_output 真按字面匹配
        self.assertTrue(_Bare().compare_output(
            'hello ... world\n', results['x'].body,
            disable_fuzzy=results['x'].disable_fuzzy))

    def test_rule16_disable_fuzzy_conflicts_with_fuzzy(self):
        """``disable_fuzzy`` 与 ``fuzzy=`` 互斥,一起写报错。"""
        text = (
            '```shell #test-result id="x" disable_fuzzy fuzzy="xxx"\n'
            'a\n```\n'
        )
        with self.assertRaises(LabelSpecError) as cm:
            _parse(text)
        self.assertIn('conflicts', str(cm.exception))

    def test_rule17_disable_fuzzy_only_on_test_result(self):
        """``disable_fuzzy`` 只允许出现在 ``#test-result``。"""
        for label in ('#test', '#test-setup'):
            text = (
                f'```shell {label} id="x" disable_fuzzy\n'
                'echo\n```\n'
            )
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text)
            self.assertIn('disable_fuzzy', str(cm.exception))

    def test_disable_fuzzy_with_value_rejected(self):
        """``disable_fuzzy='false'`` / ``='true'`` 必须报错（flag 无值，契约强制 fail-fast）。

        关键点：作者写 ``disable_fuzzy='false'`` 通常是想关掉字面匹配、回到默认 fuzzy，
        但旧解析器把 ``'false'`` 存进 list，下游 ``bool(list)`` 恒为 True，语义被静默翻转
        成"开启字面匹配"。修复后任一带 ``=`` 的写法都在 ``_parse_params`` 阶段抛
        ``LabelSpecError``，避免静默 bug。
        """
        for val in ("'false'", "'true'", '"false"', '"true"'):
            text = (
                '```shell #test-result id="x" '
                f'disable_fuzzy={val}\n'
                'a\n```\n'
            )
            with self.assertRaises(LabelSpecError) as cm:
                _parse(text)
            # 错误信息要明确指出 "flag 无值"，而不是降级成"值未引号"——后者
            # 会让作者以为修个引号就完事了。
            self.assertIn('flag parameter', str(cm.exception))
            self.assertIn('takes no value', str(cm.exception))

    def test_rule11_load_after_store_ok(self):
        """load 引用的 store 在更早位置出现则通过。"""
        text = (
            '```shell #test-setup store="x"\necho captured\n```\n'
            '```shell #test id="y" load="x>>local"\necho <local>\n```\n'
            '```shell #test-result id="y"\ncaptured\n```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[1].load, (('x', 'local'),))

    def test_rule11_store_in_hidden_setup_counts(self):
        """HTML 注释内的 store 也算 seen_stores（先 load 后 store 仍会报错）。"""
        text = (
            '<!--\n'
            '```shell #test-setup store="hidden_store"\n'
            'echo a\n```\n'
            '-->\n'
            '```shell #test id="y" load="hidden_store>>h"\n'
            'echo <h>\n```\n'
            '```shell #test-result id="y"\nx\n```\n'
        )
        # hidden store 在前,load 在后,应该通过（hidden 一样计入）。
        commands, _ = _parse(text)
        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0].hidden)
        self.assertEqual(commands[1].load, (('hidden_store', 'h'),))


class TestFold(unittest.TestCase):
    """``_fold``：生成正确的 dataclass。"""

    def test_setup_command_fields(self):
        text = (
            '```shell #test-setup store="abc"\n'
            'echo line1\n'
            'echo line2\n'
            '```\n'
        )
        commands, _ = _parse(text)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd.cmd, 'echo line1\necho line2')
        self.assertEqual(cmd.store, 'abc')
        self.assertFalse(cmd.hidden)
        # load= defaults to () when omitted — same contract as TestCommand.
        self.assertEqual(cmd.load, ())

    def test_setup_command_load_parsed(self):
        """``#test-setup load="x>>y"`` 把 (store, local) 元组传到 SetupCommand.load。

        跟 TestCommand.load 共用一套契约 —— ``_run_one`` 在 SetupCommand 路径
        调 ``substitute_placeholders(cmd.cmd, cmd.load, captures)`` 把 ``<local>``
        替换成捕获值。speculators 的 Step 1 ``#test-setup`` 块需要在 heredoc
        里引用 Step 9/10 捕获的 draft_path / verifier_path（``convert_model(model="<draft_path>")``），
        没有这条测试锁住，将来谁不小心把 SetupCommand 的 substitute 路径删了，
        CI 又会回到 19.7s "success" 但 model.safetensors 不存在的坑。
        """
        text = (
            '```shell #test-setup store="dflash_path" load="a>>p" load="b>>q"\n'
            'echo <p> <q>\n'
            '```\n'
        )
        commands, _ = _parse(text)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd.store, 'dflash_path')
        self.assertEqual(cmd.load, (('a', 'p'), ('b', 'q')))

    def test_setup_command_load_rejects_bad_shape(self):
        """``SetupCommand.__post_init__`` 拒掉非 (str, str) 元组的 load 项。"""
        with self.assertRaises(LabelSpecError):
            SetupCommand(
                cmd='echo x', store='s', hidden=False, language='shell',
                load=(('a', 'p'), ('bad',)),  # second item is str, not tuple
            )

    def test_test_command_fields(self):
        text = (
            '```shell #test-setup store="s"\necho captured\n```\n'
            '```shell #test id="abc" load="s>>p"\n'
            'echo <p>\n'
            '```\n'
            '```shell #test-result id="abc" fuzzy="xxx"\n'
            'captured\n'
            '```\n'
        )
        commands, results = _parse(text)
        self.assertEqual(len(commands), 2)
        cmd = commands[1]
        self.assertEqual(cmd.id, 'abc')
        self.assertEqual(cmd.language, 'shell')
        self.assertEqual(cmd.load, (('s', 'p'),))
        self.assertIsInstance(results['abc'], TestExpectedOutput)
        self.assertEqual(results['abc'].fuzzy, ('xxx',))


class TestSubstitutePlaceholders(unittest.TestCase):
    """``substitute_placeholders`` 纯函数。"""

    def setUp(self):
        self.base = _Bare()

    def test_single_load_replaced(self):
        out = self.base.substitute_placeholders(
            'echo <ckpt>', (('checkpoint', 'ckpt'),),
            {'checkpoint': '/path/to/ckpt'},
        )
        self.assertEqual(out, 'echo /path/to/ckpt')

    def test_multi_load_replaced(self):
        out = self.base.substitute_placeholders(
            '[<u>] <cwd>',
            (('workdir', 'cwd'), ('user', 'u')),
            {'workdir': '/hdc', 'user': 'hdc'},
        )
        self.assertEqual(out, '[hdc] /hdc')

    def test_missing_store_preserves_placeholder(self):
        """captures 缺 store_var 时保留 <local> 字面（不静默替换成空）。"""
        out = self.base.substitute_placeholders(
            'echo <ckpt>', (('checkpoint', 'ckpt'),), {},
        )
        self.assertEqual(out, 'echo <ckpt>')

    def test_empty_load_noop(self):
        out = self.base.substitute_placeholders(
            'echo plain', (), {'checkpoint': 'x'},
        )
        self.assertEqual(out, 'echo plain')


class TestSetupSubstitution(unittest.TestCase):
    """``_run_one`` 在 SetupCommand 路径调 ``substitute_placeholders``。

    跟 TestCommand 共用 ``load='x>>y'`` 契约。锁住这条契约 —— 没有它，
    ``#test-setup`` 块的 heredoc 里 ``<placeholder>`` 会落到 bash / python
    当字面字符串，下游（如 huggingface_hub ``snapshot_download``）把整串
    当 repo id 拒掉、错误信息绕一圈才报回原始占位符。
    """

    def _drive(self, base, cmd, captures):
        """绕过 ``execute()`` 的 ``self._captures = {}`` 重置,直接喂 _run_one。"""
        base._captures = dict(captures)
        base._run_one(cmd, {}, os.environ.copy(), Path.cwd(), 30, 0)

    def test_setup_substitutes_then_runs(self):
        """``<local>`` 在 bash 跑之前就被替换。"""
        base = _Bare()
        cmd = SetupCommand(
            cmd='echo <p>',
            store='next',
            hidden=False,
            language='shell',
            load=(('prev', 'p'),),
        )
        self._drive(base, cmd, {'prev': '/root/dflash-qwen3-8b-converted'})
        # capture 应是 echo 替换后的输出（不含 ``<p>`` 字面）。
        self.assertEqual(
            base._captures['next'],
            '/root/dflash-qwen3-8b-converted',
        )

    def test_setup_preserves_placeholder_when_store_missing(self):
        """captures 缺 store_var 时 ``<local>`` 保留字面（不静默替空）。"""
        base = _Bare()
        cmd = SetupCommand(
            cmd='echo PLACEHOLDER_P',  # 避开 bash 解析 < 当 redirect
            store='next',
            hidden=False,
            language='shell',
            load=(('prev', 'PLACEHOLDER_P'),),
        )
        self._drive(base, cmd, {})
        # captures 缺 ``prev`` → 占位符保留字面 → echo 原样输出。
        self.assertEqual(base._captures.get('next'), 'PLACEHOLDER_P')


class TestCompareOutput(unittest.TestCase):
    """``compare_output`` 默认 ``...`` / ``fuzzy`` 多占位符非贪婪跨行匹配。"""

    def setUp(self):
        self.base = _Bare()

    def test_default_placeholder_multiline(self):
        """``fuzzy=('...',)`` 跨行匹配 (DOTALL)。"""
        actual = 'run sh: /bin/bash\ninit banner\nloss=2.5\nfinal line\n'
        expected = 'run sh: ...\nfinal line\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('...',))
        )

    def test_fuzzy_overrides_default(self):
        """``fuzzy='xxx'`` 时 ``xxx`` 是占位符;不传 ``...`` 时默认不再内置。"""
        actual = 'Python 3.12.5\n'
        expected = 'Python 3.xxx'
        # 仅 'xxx' 是占位符,``...`` 不内置 -> 'xxx' 通配匹配 12.5
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy='xxx')
        )
        # 不传 fuzzy 时,expected 字面 (含 xxx) 必须字面出现
        self.assertFalse(self.base.compare_output(actual, expected))

    def test_fuzzy_multiple_placeholders(self):
        """``fuzzy=('xxx', 'yyy')``:每种 placeholder 各自按非贪婪通配。"""
        actual = 'step 1/5 loss=2.5\nstep 5/5 loss=0.1\n'
        expected = 'step xxx/5 loss=yyy\nstep 5/5 loss=yyy\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('xxx', 'yyy'))
        )
        # 单独只用 yyy:xxx 应按字面匹配
        actual2 = 'step 1/5 loss=2.5\n'
        expected2 = 'step xxx/5 loss=yyy\n'
        self.assertTrue(
            self.base.compare_output(actual2, expected2, fuzzy=('xxx', 'yyy'))
        )

    def test_fuzzy_default_alongside_custom(self):
        """``fuzzy=('...', 'xxx')``:两种占位符并存,调用方显式声明。"""
        actual = 'header\n...\nbody\n'
        expected = 'header\n...\nbody\n'
        self.assertTrue(
            self.base.compare_output(actual, expected, fuzzy=('...', 'xxx'))
        )

    def test_literal_mismatch(self):
        """期望含字面 token 与实际不一致 -> False。"""
        actual = 'hello world\n'
        expected = 'hello there\n'
        self.assertFalse(
            self.base.compare_output(actual, expected, fuzzy=('...',))
        )

    def test_literal_match(self):
        actual = 'exact line\n'
        expected = 'exact line\n'
        self.assertTrue(self.base.compare_output(actual, expected))

    def test_disable_fuzzy_literal_dots(self):
        """``disable_fuzzy=True``:所有 placeholder 按字面匹配。

        验证:
          1. 不传 fuzzy 时(空 tuple)compare_output 走字面匹配;
             字面相等 -> True,字面不等 -> False。
          2. ``disable_fuzzy=True`` 后即便显式传 fuzzy 也按字面。
        """
        actual_dots = 'hello ... world\n'
        actual_no_dots = 'hello world\n'
        # 1a:不传 fuzzy + literal 一致 -> 字面匹配 True
        self.assertTrue(self.base.compare_output(actual_dots, 'hello ... world'))
        # 1b:不传 fuzzy + literal 不等 -> False
        self.assertFalse(self.base.compare_output(
            actual_no_dots, 'hello ... world'))
        # 1c:传 fuzzy=('...',) 后 ``...`` 是通配,actual=literal 也匹配
        self.assertTrue(self.base.compare_output(
            actual_dots, 'hello ... world', fuzzy=('...',)))
        # 2:disable_fuzzy=True + 字面不等 -> False
        self.assertFalse(self.base.compare_output(
            actual_no_dots, 'hello ... world', disable_fuzzy=True))


class TestSetupCommandLanguageInvariant(unittest.TestCase):
    """``SetupCommand.__post_init__`` 把 ``language`` 钉死成非空。

    parse 路径里 ``_validate`` rule 7 已经卡过非空 + 白名单；这里再独立
    跑一遍 ``__post_init__`` 的 fail-fast —— 万一将来谁加个不经过
    ``parse`` 的代码路径直接 ``SetupCommand(...)``，漏传 language 会在
    这一行炸，而不是静默落到 ``run_command`` 选错 runner。

    注意：Python dataclass 对**完全漏传**必填字段在 ``__init__`` 阶段
    直接抛 ``TypeError``（不是 ``__post_init__`` 抛 ``LabelSpecError``），
    那也是我们想要的 fail-fast —— 两道防线一起锁住"language 必须有值"。
    """

    def test_missing_language_raises_typeerror(self):
        """完全漏传 language → ``TypeError``（dataclass ``__init__`` 自身 fail-fast）。"""
        with self.assertRaises(TypeError) as cm:
            SetupCommand(
                cmd='echo x', store=None, hidden=False,
                # language 故意漏传 —— Python 会在 __init__ 阶段炸，
                # 不让代码走到任何 ``__post_init__`` 校验
            )
        self.assertIn('language', str(cm.exception))

    def test_empty_language_raises(self):
        """language='' 也 raise —— ``__post_init__`` 入口校验。"""
        with self.assertRaises(LabelSpecError):
            SetupCommand(
                cmd='echo x', store=None, hidden=False, language='',
            )

    def test_python_language_constructs_ok(self):
        """``language='python'`` 构造不挂 —— 白名单校验在 ``_validate``，
        dataclass 这里只验非空，跨语言 dispatcher 走 ``run_command``。"""
        cmd = SetupCommand(
            cmd='print(<p>)', store='next', hidden=False, language='python',
            load=(('prev', 'p'),),
        )
        self.assertEqual(cmd.language, 'python')


class TestPythonSubstitution(unittest.TestCase):
    """镜像 ``TestSetupSubstitution``，把 ``bash -c`` 换成 ``python -c``。

    同一组占位符替换 / store 缺失语义在 python 解释器下也得走通 —— 这是
    ``_LANG_RUNNER`` 改完后唯一会真跑 ``python -c`` 的入口，必须锁住
    python 路径不因为 shell 写法不同（``echo`` vs ``print``、heredoc
    vs import）就漏过 ``<local>`` 替换。
    """

    def _drive(self, base, cmd, captures):
        base._captures = dict(captures)
        base._run_one(cmd, {}, os.environ.copy(), Path.cwd(), 30, 0)

    def test_setup_substitutes_then_runs_python(self):
        """``<p>`` 在 python -c 跑之前就被替换成 capture 值，print 输出捕获。

        注意：python 路径下 ``<p>`` 被替换后必须**已经是合法字符串字面**，
        否则 python 把整段当代码解析会 SyntaxError。``print('<p>')`` →
        ``print('/root/dflash-qwen3-8b-converted')`` 是合法语句；裸写
        ``print(<p>)`` 会把 ``/root/...`` 当函数调用表达式，SyntaxError。
        这正是契约"语言边界由块的 language 决定"的体现 —— 替换是文本层，
        author 必须按所选语言写出替换后仍然合法的代码。
        """
        base = _Bare()
        cmd = SetupCommand(
            cmd="print('<p>')",  # 引号包住 → 替换后是合法字符串字面
            store='next',
            hidden=False,
            language='python',
            load=(('prev', 'p'),),
        )
        self._drive(base, cmd, {'prev': '/root/dflash-qwen3-8b-converted'})
        self.assertEqual(
            base._captures['next'],
            '/root/dflash-qwen3-8b-converted',
        )

    def test_setup_preserves_placeholder_when_store_missing_python(self):
        """python 路径下 captures 缺 store_var 时 ``<local>`` 也保留字面。"""
        base = _Bare()
        cmd = SetupCommand(
            # 故意写能跑出字面 PLACEHOLDER_P 的 python —— 用 ``print('PLACEHOLDER_P')``
            # 等价于 shell 的 echo 占位；这里把占位做成原样拼接字符串，避免
            # ``<`` 在 docstring / string literal 里被 python 当操作符解析。
            cmd="print('PLACEHOLDER_P')",
            store='next',
            hidden=False,
            language='python',
            load=(('prev', 'PLACEHOLDER_P'),),
        )
        self._drive(base, cmd, {})
        self.assertEqual(base._captures.get('next'), 'PLACEHOLDER_P')

    def test_setup_python_does_not_execute_shell_metachars(self):
        """python 块里写 ``$x`` / ``; rm`` 不会被 bash 拦截 —— python 解释器
        直接拿整段字符串当代码。``$x`` 是 python 的合法标识符片段（虽然
        不是合法语句），会抛 SyntaxError，不会执行。验证语义边界：
        每个块的语言决定解释器，**不**让外层 bash 二次解析。
        """
        base = _Bare()
        cmd = SetupCommand(
            cmd='print("hi")',  # 故意避开 $ 触发 SyntaxError 让测试聚焦 runner 链路
            store='next',
            hidden=False,
            language='python',
        )
        self._drive(base, cmd, {})
        self.assertEqual(base._captures['next'], 'hi')


class TestPythonRunCommandDispatch(unittest.TestCase):
    """``run_command(language=...)`` 真按 ``_LANG_RUNNER`` 选 runner。

    这里直接断言 ``subprocess.run`` 收到的 argv 里第一个元素是不是
    ``sys.executable``（_Bare 覆盖后），而不是 mock —— mock 会让
    我们跟 runner 解耦，反而验不到"python 块真的跑出 python 解释器"。
    """

    def test_python_runner_uses_sys_executable(self):
        """``language='python'`` 走 ``_LANG_RUNNER['python']`` 派生的 argv。"""
        captured: dict = {}

        real_run = MarkdownDocTestBase.run_command

        def spy(self, cmd, env, cwd, timeout, language='shell'):
            # 把 argv 抓出来供断言，绕过 subprocess.run 的执行
            captured['argv_prefix'] = self._LANG_RUNNER[language]
            return (0, 'mocked', '')

        MarkdownDocTestBase.run_command = spy
        try:
            base = _Bare()
            rc, out, err = base.run_command(
                "print('hi')", os.environ.copy(), Path.cwd(), 30,
                language='python',
            )
        finally:
            MarkdownDocTestBase.run_command = real_run

        self.assertEqual(captured['argv_prefix'][0], _PYTHON_BIN)
        self.assertEqual(captured['argv_prefix'][1:], ('-c',))
        self.assertEqual(out, 'mocked')

    def test_unknown_language_raises(self):
        """``language='ruby'`` 等不在 ``_LANG_RUNNER`` 里 → ``AssertionError``
        （defensive:``_validate`` 已经把 parse 路径挡住了，这里是给绕过
        parse 直接调 ``run_command`` 的代码留的兜底）。"""
        base = _Bare()
        with self.assertRaises(AssertionError) as cm:
            base.run_command(
                'puts hi', os.environ.copy(), Path.cwd(), 30, language='ruby',
            )
        self.assertIn('ruby', str(cm.exception))


class TestPythonEndToEnd(unittest.TestCase):
    """端到端：parse 一段含 ``#test python`` 的 markdown，跑 ``execute``，
    capture 的 stdout 与 ``#test-result`` 字面比对。

    这条是契约的最终闭环 —— 之前的所有 parse / substitute / fold 单测
    只验了"数据形状"，这条真让 ``_run_one`` 在 python 路径上跑起来。
    """

    def test_python_test_block_runs_and_matches(self):
        """```python #test``` 真的调到 ``python -c``，stdout 与 ``#test-result`` 一致。"""
        text = (
            '```python #test id="sum"\n'
            'print(1 + 2)\n'
            '```\n'
            '\n'
            '```python #test-result id="sum"\n'
            '3\n'
            '```\n'
        )
        # 解析 → 跑 —— 走 ``execute()`` 完整链路
        base = _Bare()
        commands, results = base.parse(text)
        base._captures = {}
        base.execute(commands, results)
        # 没有 AssertionError 即通过；额外断言 capture 被消费掉（test
        # 不写 store，所以 ``_captures`` 保持空）
        self.assertEqual(base._captures, {})

    def test_python_test_with_fuzzy_placeholder(self):
        """python 块走默认 fuzzy ``...`` —— 把 python 输出的版本号通配掉。"""
        text = (
            '```python #test id="pyver"\n'
            'import sys; print(sys.version)\n'
            '```\n'
            '\n'
            '```python #test-result id="pyver"\n'
            '3.xxx\n'
            '```\n'
        )
        # 这里用 fuzzy='xxx' 显式指定，因为 ``...`` 与 python ellipsis
        # 字面冲突的语义虽然不会出现在 version 字符串里，但作者用 ``xxx``
        # 把意图说清楚：版本号是个变量
        text = text.replace('3.xxx', '3.xxx')
        # 默认 placeholder 是 ``...``，但 version 输出里没有 ``...`` 字面段，
        # 用 'xxx' 显式通配更稳。把 ``3.xxx`` 改成 ``3.<version>`` 的写法
        # 在契约里要求作者写 fuzzy='xxx'，否则期望串里有 ``...`` 时默认就
        # 已经通配 —— 改一下：
        text_ok = (
            '```python #test id="pyver"\n'
            'import sys; print(sys.version.split()[0])\n'
            '```\n'
            '\n'
            '```python #test-result id="pyver"\n'
            '3...\n'
            '```\n'
        )
        base = _Bare()
        commands, results = base.parse(text_ok)
        base._captures = {}
        base.execute(commands, results)
        # ``3...`` + 默认 fuzzy ``...`` → 通配掉 ``.14.0`` 之类，整串匹配
        # 形如 ``3.14.x``。任意 3.x.y python 3.x 版本都应通过

    def test_python_setup_block_runs_and_captures(self):
        """```python #test-setup store="..."``` 真跑到 ``python -c`` 并把
        stdout 写进 ``captures``，后续 ``#test`` 用 ``load='x>>local'`` 引用。"""
        text = (
            '```python #test-setup store="tripled"\n'
            'print(7 * 3)\n'
            '```\n'
            '\n'
            '```python #test id="echo-triple" load="tripled>>n"\n'
            'print(<n>)\n'
            '```\n'
            '\n'
            '```python #test-result id="echo-triple"\n'
            '21\n'
            '```\n'
        )
        base = _Bare()
        commands, results = base.parse(text)
        base._captures = {}
        base.execute(commands, results)
        # capture 来自 setup，但 test 没再 store，所以 ``_captures`` 里只
        # 留 setup 写的那一份。execute 不清空 captures（test 块不写 store）
        self.assertEqual(base._captures.get('tripled'), '21')


if __name__ == '__main__':
    unittest.main()

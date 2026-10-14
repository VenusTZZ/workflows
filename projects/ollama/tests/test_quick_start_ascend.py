"""Build the engine-selected Ollama release with CANN and run the user guide.

MONITORED_DOC_URL fetches the document at the triggering commit through the
base class. UPSTREAM_REF must be a release tag; the source, embedded version
and running server must agree with it. NPU_READY=true enables the hardware
test. Waiting, cache validation and process-group cleanup belong here; the
guide starts a normal foreground server in a separate terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from workflows.markdown_doc_test_base import MarkdownDocTestBase, SetupCommand, TestCommand

PROJECT = Path(__file__).resolve().parents[1]

MODEL_FILE = 'qwen2.5-0.5b-instruct-q4_0.gguf'
MODEL_SHA256 = '7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed'
MODEL_SIZE = 428730208
MODEL_NAME = 'qwen2.5-ascend'
BASE_URL = 'http://127.0.0.1:11434'


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def valid_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            return False
    return sha256(path) == MODEL_SHA256


def release_version(ref: str) -> str:
    if not re.fullmatch(r'v\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.-]+)?', ref):
        raise AssertionError(f'Expected an official release tag, got {ref!r}')
    return ref.removeprefix('v')


def inference_evidence(response: dict, loaded: dict, log: str) -> dict:
    """Reject empty generation, CPU fallback and partial model offload."""
    allowed = {MODEL_NAME, f'{MODEL_NAME}:latest'}
    if response.get('model') not in allowed or response.get('done') is not True:
        raise AssertionError(f'Unexpected chat response: {response!r}')
    message = response.get('message', {})
    if message.get('role') != 'assistant' or not message.get('content', '').strip():
        raise AssertionError('Chat returned no assistant content')
    if response.get('eval_count', 0) <= 0 or response.get('eval_duration', 0) <= 0:
        raise AssertionError('Chat returned no measured token generation')
    models = [item for item in loaded.get('models', []) if item.get('name') in allowed]
    if not models or models[0].get('size_vram', 0) <= 0:
        raise AssertionError('The target model has no device memory in /api/ps')
    starts = list(re.finditer(r'load_tensors:\s+loading model tensors', log))
    latest = log[starts[-1].start():] if starts else log
    offloads = list(re.finditer(r'offloaded\s+(\d+)/(\d+)\s+layers to GPU', latest))
    if not offloads:
        raise AssertionError('No model offload was reported for the latest load')
    offload = offloads[-1]
    used, total = map(int, offload.groups())
    buffers = re.findall(
        r'CANN\d+\s+model buffer size\s*=\s*([\d.]+)\s*MiB', latest[offload.end():],
    )
    if not any(float(value) > 0 for value in buffers):
        raise AssertionError('No positive CANN model buffer was reported')
    if used != total or total <= 0:
        raise AssertionError('No complete model offload was reported')
    return {
        'model': response['model'], 'eval_count': response['eval_count'],
        'eval_duration': response['eval_duration'], 'size_vram': models[0]['size_vram'],
        'cann_model_buffer_mib': buffers, 'offloaded_layers': [used, total],
    }


def ensure_go(cache: Path) -> None:
    """Provision a documented prerequisite in the stripped-down runner image."""
    executable = shutil.which('go')
    if executable:
        result = subprocess.run([executable, 'version'], capture_output=True, text=True, check=True)
        match = re.search(r'go(\d+)\.(\d+)', result.stdout)
        if match and tuple(map(int, match.groups())) >= (1, 26):
            return
    arch = {'aarch64': 'arm64', 'arm64': 'arm64', 'x86_64': 'amd64'}.get(platform.machine())
    if platform.system() != 'Linux' or not arch:
        raise RuntimeError('The CANN quick start requires Linux arm64 or amd64')
    version = 'go1.26.9'
    target = cache / 'toolchains' / f'{version}-{arch}'
    if not (target / 'go' / 'bin' / 'go').is_file():
        with urllib.request.urlopen('https://go.dev/dl/?mode=json&include=all', timeout=60) as response:
            releases = json.load(response)
        package = next(
            item for release in releases if release['version'] == version
            for item in release['files']
            if item['os'] == 'linux' and item['arch'] == arch and item['kind'] == 'archive'
        )
        archive = cache / 'downloads' / package['filename']
        if not archive.is_file() or sha256(archive) != package['sha256']:
            partial = archive.with_suffix('.part')
            with urllib.request.urlopen('https://go.dev/dl/' + package['filename'], timeout=120) as response:
                with partial.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if sha256(partial) != package['sha256']:
                partial.unlink()
                raise AssertionError('Go archive checksum mismatch')
            partial.replace(archive)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(target, filter='data')
    os.environ['PATH'] = str(target / 'go' / 'bin') + os.pathsep + os.environ['PATH']


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'

    def prepare_environment(self) -> None:
        self.original_cwd = Path.cwd()
        self.original_env = os.environ.copy()
        self.work = Path(tempfile.mkdtemp(prefix='ollama-quick-start-'))
        self.server = None
        self.server_log = None
        self.addCleanup(self.post_process)
        self.ref = os.environ.get('UPSTREAM_REF', '')
        self.version = release_version(self.ref)
        self.cache = Path(os.environ.get('OLLAMA_TEST_CACHE', '/ollama-cache')).resolve()
        for name in ('downloads', 'models', 'toolchains', 'go-build', 'go-mod', 'ccache'):
            (self.cache / name).mkdir(parents=True, exist_ok=True)
        env = subprocess.run(
            ['bash', '-c', 'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null && env -0'],
            capture_output=True, check=True,
        )
        for item in env.stdout.decode().split('\0'):
            if '=' in item:
                key, value = item.split('=', 1)
                os.environ[key] = value
        if not os.environ.get('ASCEND_HOME_PATH'):
            raise AssertionError('CANN did not set ASCEND_HOME_PATH')
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
        # Prevent a pre-existing source override from changing the pinned
        # llama.cpp revision while the report still names LLAMA_CPP_VERSION.
        os.environ.pop('OLLAMA_LLAMA_CPP_SOURCE', None)
        os.environ['OLLAMA_HOST'] = BASE_URL
        os.environ['OLLAMA_DEBUG'] = '1'
        os.environ['OLLAMA_KEEP_ALIVE'] = '5m'
        for name in ('NO_PROXY', 'no_proxy'):
            os.environ[name] = ','.join(filter(None, (os.environ.get(name), '127.0.0.1', 'localhost')))
        os.environ['GOCACHE'] = str(self.cache / 'go-build')
        os.environ['GOMODCACHE'] = str(self.cache / 'go-mod')
        os.environ['CCACHE_DIR'] = str(self.cache / 'ccache')
        os.environ.setdefault('GOPROXY', 'https://goproxy.cn,direct')
        os.environ.setdefault('GOSUMDB', 'sum.golang.org https://sum.golang.google.cn')
        ensure_go(self.cache)
        for tool in ('git', 'curl', 'gcc', 'g++', 'make'):
            if not shutil.which(tool):
                raise AssertionError(f'Missing build prerequisite: {tool}')
        # This performs the guide's explicit manual patch-download step, using
        # bytes from the triggering checkout rather than a moving branch URL.
        shutil.copyfile(PROJECT / 'patches' / 'ollama-cann.patch', self.work / 'ollama-cann.patch')
        cached = self.cache / 'downloads' / MODEL_FILE
        if cached.exists() and not valid_model(cached):
            cached.unlink()
            self.log('cache: removed corrupt GGUF')
        if cached.exists():
            (self.work / MODEL_FILE).symlink_to(cached)
        for blob in (self.cache / 'models' / 'blobs').glob('sha256-*'):
            if blob.is_file() and sha256(blob) != blob.name.removeprefix('sha256-'):
                blob.unlink()
                self.log(f'cache: removed corrupt Ollama blob {blob.name}')
        (self.work / 'ollama-models').symlink_to(self.cache / 'models', target_is_directory=True)
        (self.work / 'reports').mkdir()
        self.report = {'upstream_ref': self.ref, 'native_build_verified': False, 'npu_inference_verified': False}
        os.chdir(self.work)

    def pin_source(self) -> None:
        source = self.work / 'ollama'
        for args in (
            ['fetch', '--depth', '1', 'origin', f'refs/tags/{self.ref}'],
            ['checkout', '--detach', 'FETCH_HEAD'],
        ):
            subprocess.run(['git', '-C', str(source), *args], check=True, timeout=120)
        head = subprocess.run(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, encoding='utf-8', check=True, timeout=120,
        )
        patch = self.work / 'ollama-cann.patch'
        checked = subprocess.run(
            ['git', '-C', str(source), 'apply', '--check', str(patch)],
            capture_output=True, text=True, encoding='utf-8', timeout=120,
        )
        if checked.returncode:
            raise AssertionError(f'CANN patch incompatible with {self.ref}:\n{checked.stderr}')
        self.report.update({
            'source_commit': head.stdout.strip(),
            'patch_sha256': sha256(patch),
            'llama_cpp_ref': (source / 'LLAMA_CPP_VERSION').read_text(encoding='utf-8').strip(),
            'patch_applicable': True,
        })
        self.write_report()

    def write_report(self) -> None:
        body = json.dumps(self.report, indent=2, ensure_ascii=False)
        (self.work / 'reports' / 'ollama-compatibility.json').write_text(body + '\n', encoding='utf-8')
        self.log('Ollama verification evidence:\n' + body)

    def server_tail(self) -> str:
        path = self.work / 'reports' / 'server.log'
        return path.read_text(encoding='utf-8', errors='replace')[-16000:] if path.exists() else '(no server log)'

    def request(self, path: str) -> dict:
        if self.server is None or self.server.poll() is not None:
            raise AssertionError('Ollama server exited\n' + self.server_tail())
        # Local service probes must not go through the runner's HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(BASE_URL + path, timeout=5) as response:
            return json.load(response)

    def start_service(self, command: str, env: dict, cwd: Path) -> None:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', 11434))
        self.server_log = (self.work / 'reports' / 'server.log').open('wb')
        self.server = subprocess.Popen(
            ['bash', '-c', 'exec env ' + command], env=env, cwd=cwd,
            stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 120
        last_error = ''
        while time.monotonic() < deadline:
            try:
                actual = self.request('/api/version')
                if actual.get('version') != self.version:
                    raise AssertionError(f'Built/server version mismatch: {actual} != {self.ref}')
                self.report['server_version'] = actual['version']
                self.write_report()
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise AssertionError(f'Ollama not ready after 120s: {last_error}\n{self.server_tail()}')

    def run_command(self, cmd, env, cwd, timeout, language='shell'):
        rc, out, err = super().run_command(cmd, env, cwd, timeout, language)
        if './ollama/ollama run ' in cmd and rc == 0:
            if not re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', out).strip():
                raise AssertionError('CLI returned no generated content')
            self.log_block('CLI metrics', err.splitlines(), cap=0)
        return rc, out, err

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand) and './ollama/ollama serve' in cmd.cmd:
            self.start_service(cmd.cmd, env, cwd)
            return
        try:
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            if cmd.cmd.startswith('git clone '):
                self.pin_source()
            elif cmd.cmd.startswith('git -C ollama apply '):
                self.report['patch_applied_by_this_run'] = True
                self.write_report()
            elif cmd.cmd.startswith('cmake --build '):
                source = self.work / 'ollama'
                payload = source / 'build' / 'lib' / 'ollama'
                for path in (source / 'ollama', payload / 'cann' / 'libggml-cann.so', payload / 'llama-server'):
                    if not path.is_file():
                        raise AssertionError(f'Missing build product: {path}')
                subprocess.run(
                    ['go', 'test', './discover', './ml', '-run', 'TestCANN', '-count=1', '-v'],
                    cwd=source, env=env, check=True, timeout=timeout,
                )
                self.report['native_build_verified'] = True
                self.write_report()
                probe_env = env.copy()
                probe_env['GGML_BACKEND_PATH'] = str(payload / 'cann' / 'libggml-cann.so')
                probe_env['LD_LIBRARY_PATH'] = ':'.join(filter(None, (
                    str(payload), str(payload / 'cann'), env.get('LD_LIBRARY_PATH'),
                )))
                probe = subprocess.run(
                    [str(payload / 'llama-server'), '--list-devices', '--offline'],
                    env=probe_env, cwd=cwd, capture_output=True, text=True, timeout=120, check=True,
                )
                self.log_block('CANN device probe', (probe.stdout + probe.stderr).splitlines(), cap=0)
                if not re.search(r'CANN\d+:', probe.stdout):
                    raise AssertionError('Built runner did not enumerate a CANN device')
                self.report['runtime_load_verified'] = True
                self.write_report()
            elif cmd.language == 'python' and "model = Path('" + MODEL_FILE in cmd.cmd:
                model = self.work / MODEL_FILE
                if not valid_model(model):
                    model.unlink(missing_ok=True)
                    raise AssertionError('Downloaded GGUF checksum or size mismatch')
                cached = self.cache / 'downloads' / MODEL_FILE
                if not model.is_symlink():
                    shutil.move(str(model), cached)
                    model.symlink_to(cached)
                self.report['model_sha256'] = MODEL_SHA256
            elif isinstance(cmd, TestCommand) and cmd.id == 'api-chat':
                response = json.loads((self.work / 'chat-response.json').read_text(encoding='utf-8'))
                shutil.copyfile(self.work / 'chat-response.json', self.work / 'reports' / 'chat-response.json')
                loaded = self.request('/api/ps')
                (self.work / 'reports' / 'loaded-models.json').write_text(json.dumps(loaded), encoding='utf-8')
                log = (self.work / 'reports' / 'server.log').read_text(encoding='utf-8', errors='replace')
                self.report['inference'] = inference_evidence(response, loaded, log)
                self.report['npu_inference_verified'] = True
                self.write_report()
        except Exception:
            self.log('Ollama service log tail:\n' + self.server_tail())
            raise

    def post_process(self) -> None:
        if getattr(self, '_cleaned', False):
            return
        self._cleaned = True
        process = getattr(self, 'server', None)
        try:
            if process is not None:
                # The runner may outlive the server leader; always signal the group.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.2)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=10)
            handle = getattr(self, 'server_log', None)
            if handle is not None:
                handle.close()
            if hasattr(self, 'cache') and (self.work / 'reports').exists():
                response = self.work / 'chat-response.json'
                if response.exists():
                    shutil.copyfile(response, self.work / 'reports' / response.name)
                saved = self.cache / 'reports' / self.work.name
                shutil.copytree(self.work / 'reports', saved, dirs_exist_ok=True)
                self.log(f'Full reports preserved on the bind-mounted cache: {saved}')
        finally:
            if hasattr(self, 'original_cwd'):
                os.chdir(self.original_cwd)
                os.environ.clear()
                os.environ.update(self.original_env)
                shutil.rmtree(self.work)

    @unittest.skipUnless(os.environ.get('NPU_READY', '').lower() == 'true', 'requires an Ascend NPU runner')
    def test_runs_doc(self) -> None:
        self.prepare_environment()
        self.run_template()


if __name__ == '__main__':
    unittest.main()

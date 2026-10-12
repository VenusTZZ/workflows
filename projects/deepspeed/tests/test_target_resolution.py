"""Replay the engine's manual ref resolution with offline release responses."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[3]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"),
                     "ref resolution subprocess tests require POSIX bash")
class TargetResolutionTests(unittest.TestCase):
    def resolve(self, *, http: str, tag: str = "", explicit: str = ""):
        caller = (REPO / ".github/workflows/deepspeed-examples.yml").read_text()
        default = re.search(r"^      default_branch: (\S+)$", caller, re.MULTILINE)
        self.assertIsNotNone(default, "caller must configure its fallback branch")
        engine = (REPO / ".github/workflows/examples-template.yml").read_text()
        # Execute the actual manual branch unchanged, not a reimplemented resolver.
        step = engine.split("      - name: Decide target\n", 1)[1]
        manual = textwrap.dedent(step.split("        run: |\n", 1)[1]
                                 .split('          NEED="', 1)[0])
        mock = r'''
curl() {
  local output=''
  while (($#)); do
    if [[ "$1" == -o ]]; then output="$2"; shift; fi
    shift
  done
  printf '%s' "$RELEASE_JSON" > "$output"
  printf '%s' "$HTTP_CODE"
}
jq() {
  "$TEST_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("tag_name") or "")' "${@: -1}"
}
'''
        with tempfile.TemporaryDirectory(prefix="ds ref ") as tmp:
            output = Path(tmp) / "outputs"
            env = dict(os.environ, EVENT_NAME="workflow_dispatch", INPUT_REF=explicit,
                       INPUT_REPO="", UPSTREAM_REPO="deepspeedai/DeepSpeed",
                       DEFAULT_BRANCH=default.group(1), GITHUB_OUTPUT=str(output),
                       RELEASE_JSON=json.dumps({"tag_name": tag}), HTTP_CODE=http,
                       TEST_PYTHON=sys.executable)
            result = subprocess.run(["bash", "-c", mock + manual], env=env,
                                    text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(outputs["target_repo"], "deepspeedai/DeepSpeed")
            self.assertEqual(outputs["need_to_run"], "true")
            return outputs["target_ref"], result.stdout

    def test_run_26_http_403_falls_back_to_master(self):
        ref, log = self.resolve(http="403")
        self.assertEqual(ref, "master")
        self.assertIn("HTTP 403", log)
        self.assertIn("reason=branch-fallback", log)

    def test_missing_release_or_empty_tag_falls_back_to_master(self):
        for http in ("404", "200", "000"):
            with self.subTest(http=http):
                self.assertEqual(self.resolve(http=http)[0], "master")

    def test_latest_release_retains_priority(self):
        ref, log = self.resolve(http="200", tag="v0.19.7")
        self.assertEqual(ref, "v0.19.7")
        self.assertIn("reason=latest-release", log)

    def test_explicit_branch_tag_or_sha_is_preserved(self):
        for ref in ("master", "v0.19.7", "a" * 40):
            with self.subTest(ref=ref):
                resolved, log = self.resolve(http="403", explicit=ref)
                self.assertEqual(resolved, ref)
                self.assertIn("reason=explicit", log)
                self.assertNotIn("::warning::", log)


if __name__ == "__main__":
    unittest.main()

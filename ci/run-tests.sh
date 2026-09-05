#!/usr/bin/env bash
# Run the pytest suite and fail if collection/execution drops below its baseline.
set -uo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 2

junit_xml="${CI_JUNIT_XML:-wlapci-pytest.xml}"
pytest_rc=0
python -m pytest tests/ -q -p no:cacheprovider --junitxml="$junit_xml" || pytest_rc=$?

case_count="$(python - "$junit_xml" <<'PY'
import sys
from xml.etree import ElementTree

try:
    root = ElementTree.parse(sys.argv[1]).getroot()
    print(len(root.findall(".//testcase")))
except (FileNotFoundError, ElementTree.ParseError):
    print("0")
PY
)"

floor=104
if ! [[ "$case_count" =~ ^[0-9]+$ ]] || (( case_count < floor )); then
    printf 'FAIL: pytest reported %s cases; floor is %s\n' "$case_count" "$floor" >&2
    exit 1
fi

if (( pytest_rc != 0 )); then
    printf 'FAIL: pytest exited %s after reporting %s cases\n' "$pytest_rc" "$case_count" >&2
    exit "$pytest_rc"
fi

printf 'PASS: pytest reported %s cases (floor %s)\n' "$case_count" "$floor"

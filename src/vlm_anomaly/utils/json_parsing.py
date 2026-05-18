"""Robust extraction of structured JSON from free-form VLM responses.

Full implementation lives in task 06. See CLAUDE.md §7.3 for the fallback
chain (json.loads → balanced-brace extraction → code-fence stripping →
regex key extraction → parse_error=True).
"""

from __future__ import annotations

"""WP6b — knowledge-base routing refinement (pab.agent).

Covers the provider-aware PA_ROUTE modes wired into _kb_block / _kb_style / _system:
  auto (default) — anthropic/claude_cli -> full KB + naming-only note; zhipu/gemini -> subset
  PA_ROUTE=1     — force the regime subset for all providers
  PA_ROUTE=0     — force the full KB with NO note (byte-identical legacy path)
Plus: the two new universal cards are core (present in every variant), the full note carries
both the setup and principles indexes, and the _SYSTEM_CACHE key is shared where the routed
prompt is identical and distinct where it differs.

No LLM, no network: everything asserts on the assembled system string / cache keys directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pab.agent as ag  # noqa: E402

KB = Path(__file__).resolve().parents[1] / "knowledge"
NEW_CARDS = ("stop-placement-and-management", "no-trade-environments")


# ---------- helpers: reconstruct the KB the pre-change way, straight from the files ----------

def _cards_ref(sub: str):
    """(stem, text) pairs in the SAME order pab.agent._cards uses: sorted(glob('*.md'))."""
    d = KB / sub
    return [(f.stem, f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.md"))]


def _full_block_ref() -> str:
    """The full-KB block (no note) exactly as _kb_block assembled it before this change."""
    prin, setups = _cards_ref("principles"), _cards_ref("setups")
    parts = [f"### principles/{n}\n{t}" for n, t in prin]
    parts += [f"### setups/{n}\n{t}" for n, t in setups]
    return "\n\n".join(parts)


def _idx(sub: str) -> str:
    return ", ".join(sorted(n for n, _ in _cards_ref(sub)))


def _reset():
    ag._SYSTEM_CACHE.clear()
    ag._CARDS_CACHE.clear()


# ---------- (a) PA_ROUTE=0 is byte-identical to a from-files reconstruction ----------

def test_route_off_full_kb_byte_identical(monkeypatch):
    monkeypatch.setenv("PA_ROUTE", "0")
    _reset()
    ref = _full_block_ref()
    # the full-KB block reconstructed from the card files, byte-for-byte, regime-independent:
    assert ag._kb_block(None, "full") == ref
    assert ag._kb_block("trend", "full") == ref
    # ...and it is exactly what _system emits (no note), for EVERY provider (provider must
    # not leak into the PA_ROUTE=0 path -> its decision-cache keys survive unchanged):
    # NOTE: the OUTPUT v2 prose mentions the phrase "DIAGNOSIS HINT" as a reference, so the
    # no-note check keys on the actual note HEADER ("... a code pre-classifier ...").
    for prov in (None, "anthropic", "claude_cli", "zhipu", "gemini"):
        s = ag._system(False, "trend", "mes", "am", prov)
        assert s.endswith(ref), prov
        assert "DIAGNOSIS HINT: a code pre-classifier" not in s, prov
    assert (ag._system(False, "trend", "mes", "am", "anthropic")
            == ag._system(False, "trend", "mes", "am", "zhipu"))


# ---------- (b) auto mode: strong providers full+note, free-tier providers subset ----------

def test_auto_mode_provider_routing(monkeypatch):
    monkeypatch.delenv("PA_ROUTE", raising=False)   # unset -> auto (the new default)
    _reset()
    prin, setups = _cards_ref("principles"), _cards_ref("setups")
    idx_s, idx_p = _idx("setups"), _idx("principles")

    for prov in ("anthropic", "claude_cli"):        # full KB + full-KB note
        s = ag._system(False, "trend", "mes", "am", prov)
        assert "DIAGNOSIS HINT: a code pre-classifier" in s, prov
        assert "'trend'" in s and "all cards are attached" in s, prov
        assert "subset relevant to that regime" not in s, prov      # full-KB phrasing
        assert f"Full setup index: {idx_s}" in s, prov
        assert f"Full principles index: {idx_p}" in s, prov
        for n, _ in prin:                            # every principle attached
            assert f"### principles/{n}\n" in s, (prov, n)
        for n, _ in setups:                          # every setup attached
            assert f"### setups/{n}\n" in s, (prov, n)

    z = ag._system(False, "trend", "mes", "am", "zhipu")            # regime subset + note
    assert "DIAGNOSIS HINT: a code pre-classifier" in z and "subset relevant to that regime" in z
    assert f"Full setup index (exists in your KB even when not attached): {idx_s}" in z
    assert f"Full principles index: {idx_p}" in z                  # breadcrumb for unselected
    assert "### setups/channel-pullback-h1h2-l1l2\n" in z          # trend setup present
    assert "### setups/triangle\n" not in z                        # range setup omitted
    # gemini routes exactly like zhipu; both are strictly smaller than the strong-provider KB
    assert ag._system(False, "trend", "mes", "am", "gemini") == z
    assert len(z) < len(ag._system(False, "trend", "mes", "am", "anthropic"))


# ---------- (c) PA_ROUTE=1 forces the subset for claude_cli too ----------

def test_route1_forces_subset_for_claude_cli(monkeypatch):
    monkeypatch.setenv("PA_ROUTE", "1")
    _reset()
    s = ag._system(False, "trend", "mes", "am", "claude_cli")
    assert "DIAGNOSIS HINT: a code pre-classifier" in s and "subset relevant to that regime" in s
    assert "### setups/triangle\n" not in s                        # range card omitted -> subset
    assert "all cards are attached" not in s
    # provider-independent under PA_ROUTE=1: claude_cli == zhipu == anthropic
    assert s == ag._system(False, "trend", "mes", "am", "zhipu")
    assert s == ag._system(False, "trend", "mes", "am", "anthropic")


# ---------- (d) the two new cards are core -> present in EVERY variant ----------

def test_new_cards_are_core_in_every_variant(monkeypatch):
    for c in NEW_CARDS:
        assert c in ag.CORE_PRINCIPLES                            # registered as core doctrine
    variants = [
        ("0", "anthropic"),      # full, no note
        ("1", "claude_cli"),     # subset (forced)
        (None, "anthropic"),     # auto -> full_note
        (None, "zhipu"),         # auto -> subset
    ]
    for val, prov in variants:
        if val is None:
            monkeypatch.delenv("PA_ROUTE", raising=False)
        else:
            monkeypatch.setenv("PA_ROUTE", val)
        _reset()
        s = ag._system(False, "trend", "mes", "am", prov)
        for c in NEW_CARDS:
            assert f"### principles/{c}\n" in s, (val, prov, c)


# ---------- (e) _SYSTEM_CACHE keys: shared where outputs match, distinct where they differ ----

def test_system_cache_key_scheme(monkeypatch):
    # auto: 4 providers collapse to 2 distinct routed prompts -> exactly 2 cache keys
    monkeypatch.delenv("PA_ROUTE", raising=False)
    _reset()
    texts = {p: ag._system(False, "trend", "mes", "am", p)
             for p in ("anthropic", "claude_cli", "zhipu", "gemini")}
    assert texts["anthropic"] == texts["claude_cli"]              # same style -> shared slot
    assert texts["zhipu"] == texts["gemini"]
    assert texts["anthropic"] != texts["zhipu"]                  # full_note vs subset differ
    assert len(ag._SYSTEM_CACHE) == 2

    # PA_ROUTE=0: provider must NOT split the key (legacy path, byte-identical) -> 1 key
    monkeypatch.setenv("PA_ROUTE", "0")
    _reset()
    a = ag._system(False, "trend", "mes", "am", "anthropic")
    z = ag._system(False, "trend", "mes", "am", "zhipu")
    assert a == z and len(ag._SYSTEM_CACHE) == 1

    # PA_ROUTE=1: subset for all, provider-independent -> 1 key
    monkeypatch.setenv("PA_ROUTE", "1")
    _reset()
    a1 = ag._system(False, "trend", "mes", "am", "anthropic")
    z1 = ag._system(False, "trend", "mes", "am", "zhipu")
    assert a1 == z1 and len(ag._SYSTEM_CACHE) == 1


# ---------- mode/style mapping (guards the env parsing + provider classification) ----------

def test_route_mode_and_kb_style_mapping(monkeypatch):
    for val, mode in [(None, "auto"), ("auto", "auto"), ("1", "subset"), ("true", "subset"),
                      ("0", "full"), ("no", "full"), (" AUTO ", "auto")]:
        if val is None:
            monkeypatch.delenv("PA_ROUTE", raising=False)
        else:
            monkeypatch.setenv("PA_ROUTE", val)
        assert ag._route_mode() == mode, val

    monkeypatch.delenv("PA_ROUTE", raising=False)                 # auto
    assert ag._kb_style("anthropic") == "full_note"
    assert ag._kb_style("claude_cli") == "full_note"
    assert ag._kb_style("zhipu") == "subset"
    assert ag._kb_style("gemini") == "subset"
    assert ag._kb_style(None) == "subset"                        # unspecified -> non-strong
    monkeypatch.setenv("PA_ROUTE", "1")
    assert ag._kb_style("anthropic") == "subset"                # forced subset for all
    monkeypatch.setenv("PA_ROUTE", "0")
    assert ag._kb_style("zhipu") == "full"                      # forced full for all

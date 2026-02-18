"""Tests for pipeline presets."""

from __future__ import annotations

import pytest

from triad.presets import (
    DEFAULT_PRESET,
    PRESETS,
    format_prompt_tag,
    matching_preset,
    resolve_preset,
)


class TestResolvePreset:
    """Tests for resolve_preset()."""

    def test_default_preset(self):
        """No arguments → balanced preset."""
        mode, route, arbiter = resolve_preset()
        assert mode == "sequential"
        assert route == "hybrid"
        assert arbiter == "bookend"

    def test_named_preset_fast(self):
        mode, route, arbiter = resolve_preset("fast")
        assert mode == "sequential"
        assert route == "speed_first"
        assert arbiter == "off"

    def test_named_preset_explore(self):
        mode, route, arbiter = resolve_preset("explore")
        assert mode == "parallel"
        assert route == "hybrid"
        assert arbiter == "bookend"

    def test_named_preset_debate(self):
        mode, route, arbiter = resolve_preset("debate")
        assert mode == "debate"
        assert route == "quality_first"
        assert arbiter == "bookend"

    def test_named_preset_thorough(self):
        mode, route, arbiter = resolve_preset("thorough")
        assert mode == "sequential"
        assert route == "quality_first"
        assert arbiter == "full"

    def test_named_preset_cheap(self):
        mode, route, arbiter = resolve_preset("cheap")
        assert mode == "sequential"
        assert route == "cost_optimized"
        assert arbiter == "off"

    def test_preset_with_mode_override(self):
        """Flag override replaces the preset's mode."""
        mode, route, arbiter = resolve_preset("fast", mode="parallel")
        assert mode == "parallel"
        assert route == "speed_first"  # from fast preset
        assert arbiter == "off"  # from fast preset

    def test_preset_with_arbiter_override(self):
        mode, route, arbiter = resolve_preset("fast", arbiter="bookend")
        assert mode == "sequential"
        assert route == "speed_first"
        assert arbiter == "bookend"  # overridden

    def test_preset_with_route_override(self):
        mode, route, arbiter = resolve_preset("balanced", route="quality_first")
        assert mode == "sequential"
        assert route == "quality_first"  # overridden
        assert arbiter == "bookend"

    def test_preset_with_all_overrides(self):
        mode, route, arbiter = resolve_preset(
            "fast", mode="debate", route="hybrid", arbiter="full",
        )
        assert mode == "debate"
        assert route == "hybrid"
        assert arbiter == "full"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset.*'nonexistent'"):
            resolve_preset("nonexistent")

    def test_none_preset_uses_default(self):
        """Explicitly passing None uses balanced."""
        mode, route, arbiter = resolve_preset(None)
        assert (mode, route, arbiter) == PRESETS[DEFAULT_PRESET]


class TestMatchingPreset:
    """Tests for matching_preset()."""

    def test_matches_balanced(self):
        assert matching_preset("sequential", "hybrid", "bookend") == "balanced"

    def test_matches_fast(self):
        assert matching_preset("sequential", "speed_first", "off") == "fast"

    def test_no_match(self):
        assert matching_preset("sequential", "hybrid", "off") is None

    def test_all_presets_round_trip(self):
        """Every preset in the table should match itself."""
        for name, (m, r, a) in PRESETS.items():
            assert matching_preset(m, r, a) == name


class TestFormatPromptTag:
    """Tests for format_prompt_tag()."""

    def test_known_preset_shows_name(self):
        assert format_prompt_tag("sequential", "hybrid", "bookend") == "balanced"

    def test_known_preset_fast(self):
        assert format_prompt_tag("sequential", "speed_first", "off") == "fast"

    def test_custom_shows_abbreviations(self):
        tag = format_prompt_tag("sequential", "hybrid", "off")
        assert tag == "seq|hyb|off"

    def test_custom_parallel(self):
        tag = format_prompt_tag("parallel", "quality_first", "full")
        assert tag == "par|qual|full"

    def test_debate_preset(self):
        assert format_prompt_tag("debate", "quality_first", "bookend") == "debate"

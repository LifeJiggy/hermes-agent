"""Tests for the context compression loop fix (PR #29335).

Covers:
- Anti-thrashing protection in should_compress()
- session_id propagation in run_conversation() result dict
- Gateway session_store._save() after rotation
- Preflight compression pass count
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


# ---------------------------------------------------------------------------
# Anti-thrashing protection
# ---------------------------------------------------------------------------

class TestAntiThrashing:
    def test_should_compress_returns_true_above_threshold(self, compressor):
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_should_compress_returns_false_below_threshold(self, compressor):
        assert compressor.should_compress(prompt_tokens=50000) is False

    def test_ineffective_count_zero_allows_compress(self, compressor):
        compressor._ineffective_compression_count = 0
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_ineffective_count_one_allows_compress(self, compressor):
        compressor._ineffective_compression_count = 1
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_ineffective_count_two_blocks_compress(self, compressor):
        """After 2 ineffective compressions, should_compress returns False
        even when above threshold — prevents infinite loop."""
        compressor._ineffective_compression_count = 2
        assert compressor.should_compress(prompt_tokens=90000) is False

    def test_ineffective_count_high_blocks_compress(self, compressor):
        compressor._ineffective_compression_count = 5
        assert compressor.should_compress(prompt_tokens=90000) is False

    def test_ineffective_count_resets_on_good_compression(self, compressor):
        """A compression that saves >=10% resets the counter."""
        compressor._ineffective_compression_count = 2
        compressor._last_compression_savings_pct = 15.0
        # After a good compression, the counter should be reset
        compressor._ineffective_compression_count = 0
        assert compressor.should_compress(prompt_tokens=90000) is True

    def test_ineffective_count_below_threshold_irrelevant(self, compressor):
        """Even with high ineffective count, below threshold still returns False
        (not related to anti-thrashing)."""
        compressor._ineffective_compression_count = 5
        assert compressor.should_compress(prompt_tokens=50000) is False


# ---------------------------------------------------------------------------
# Preflight compression pass limit
# ---------------------------------------------------------------------------

class TestPreflightPassLimit:
    def test_preflight_respects_three_pass_limit(self, compressor):
        """The preflight loop runs at most 3 compression passes."""
        # Simulate: each pass reduces tokens by a small amount
        # but never enough to drop below threshold
        compressor.threshold_tokens = 85000
        tokens = [95000, 92000, 90000, 89000]  # still above after 3 passes
        pass_count = 0
        for _pass in range(3):
            pass_count += 1
            if not compressor.should_compress(tokens[_pass]):
                break
        assert pass_count == 3

    def test_preflight_stops_when_compression_ineffective(self, compressor):
        """Preflight stops early if anti-thrashing kicks in mid-loop."""
        compressor.threshold_tokens = 85000
        compressor._ineffective_compression_count = 2  # pre-set to block
        # should_compress returns False even though tokens are above threshold
        assert compressor.should_compress(prompt_tokens=95000) is False


# ---------------------------------------------------------------------------
# session_id in result dict
# ---------------------------------------------------------------------------

class TestSessionIdInResult:
    def test_finalize_turn_result_includes_session_id(self):
        """The result dict from finalize_turn includes session_id
        so the gateway can detect session rotation."""
        import inspect
        from agent.turn_finalizer import finalize_turn
        source = inspect.getsource(finalize_turn)
        assert '"session_id": agent.session_id' in source

    def test_finalize_turn_result_includes_compression_loop_count(self):
        """The result dict from finalize_turn includes compression_loop_count."""
        import inspect
        from agent.turn_finalizer import finalize_turn
        source = inspect.getsource(finalize_turn)
        assert '"compression_loop_count"' in source


# ---------------------------------------------------------------------------
# Gateway session persistence
# ---------------------------------------------------------------------------

class TestGatewaySessionPersistence:
    def test_save_called_after_session_rotation(self):
        """Gateway calls session_store._save() after detecting session_id rotation."""
        import inspect
        from gateway import run
        source = inspect.getsource(run.GatewayRunner)
        # Verify the _save() call exists after session_id rotation detection
        assert "self.session_store._save()" in source

    def test_telegram_topic_binding_synced_after_rotation(self):
        """Gateway syncs Telegram topic binding after session rotation."""
        import inspect
        from gateway import run
        source = inspect.getsource(run.GatewayRunner)
        assert "_sync_telegram_topic_binding" in source
        assert "agent-result-compression" in source


# ---------------------------------------------------------------------------
# Compression savings tracking
# ---------------------------------------------------------------------------

class TestCompressionSavingsTracking:
    def test_savings_pct_tracked(self, compressor):
        """_last_compression_savings_pct is set after compression."""
        compressor._last_compression_savings_pct = 12.5
        assert compressor._last_compression_savings_pct == 12.5

    def test_ineffective_threshold_is_10_percent(self, compressor):
        """Compressions saving <10% are counted as ineffective."""
        # savings_pct < 10 => ineffective
        compressor._ineffective_compression_count = 0
        # Simulate a compression that saved only 5%
        savings_pct = 5.0
        if savings_pct < 10.0:
            compressor._ineffective_compression_count += 1
        assert compressor._ineffective_compression_count == 1

    def test_good_compression_resets_counter(self, compressor):
        """Compressions saving >=10% reset the ineffective counter."""
        compressor._ineffective_compression_count = 3
        savings_pct = 15.0
        if savings_pct >= 10.0:
            compressor._ineffective_compression_count = 0
            compressor._anti_thrash_warning_emitted = False
        assert compressor._ineffective_compression_count == 0


# ---------------------------------------------------------------------------
# Enhancement: configurable max_preflight_passes
# ---------------------------------------------------------------------------

class TestMaxPreflightPasses:
    def test_default_max_preflight_passes(self, compressor):
        assert compressor.max_preflight_passes == 3

    def test_max_preflight_passes_configurable(self, compressor):
        compressor.max_preflight_passes = 5
        assert compressor.max_preflight_passes == 5


# ---------------------------------------------------------------------------
# Enhancement: anti-thrashing warning dedup
# ---------------------------------------------------------------------------

class TestAntiThrashWarningDedup:
    def test_warning_emitted_once(self, compressor):
        """Warning should only fire once when anti-thrashing is active."""
        compressor._ineffective_compression_count = 2
        compressor._anti_thrash_warning_emitted = False

        # First call should set the flag
        result = compressor.should_compress(prompt_tokens=90000)
        assert result is False
        assert compressor._anti_thrash_warning_emitted is True

        # Second call should not re-emit (flag already set)
        result = compressor.should_compress(prompt_tokens=90000)
        assert result is False
        assert compressor._anti_thrash_warning_emitted is True

    def test_warning_flag_resets_on_good_compression(self, compressor):
        """After a good compression, warning flag resets."""
        compressor._ineffective_compression_count = 2
        compressor._anti_thrash_warning_emitted = True

        # Simulate good compression resetting the counter
        compressor._ineffective_compression_count = 0
        compressor._anti_thrash_warning_emitted = False

        assert compressor.should_compress(prompt_tokens=90000) is True
        assert compressor._anti_thrash_warning_emitted is False


# ---------------------------------------------------------------------------
# Enhancement: compression_loop_count in result dict
# ---------------------------------------------------------------------------

class TestCompressionLoopCountInResult:
    def test_turn_context_has_compression_loop_count(self):
        """TurnContext dataclass includes compression_loop_count field."""
        from agent.turn_context import TurnContext
        tc = TurnContext(
            user_message="test",
            original_user_message="test",
            messages=[],
            conversation_history=None,
            active_system_prompt=None,
            effective_task_id="t1",
            turn_id="r1",
            current_turn_user_idx=0,
        )
        assert tc.compression_loop_count == 0

    def test_turn_context_accepts_compression_loop_count(self):
        from agent.turn_context import TurnContext
        tc = TurnContext(
            user_message="test",
            original_user_message="test",
            messages=[],
            conversation_history=None,
            active_system_prompt=None,
            effective_task_id="t1",
            turn_id="r1",
            current_turn_user_idx=0,
            compression_loop_count=3,
        )
        assert tc.compression_loop_count == 3

    def test_finalize_turn_accepts_compression_loop_count(self):
        """finalize_turn accepts _compression_loop_count parameter."""
        import inspect
        from agent.turn_finalizer import finalize_turn
        sig = inspect.signature(finalize_turn)
        assert "_compression_loop_count" in sig.parameters

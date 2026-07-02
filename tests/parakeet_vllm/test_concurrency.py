"""Task 9 concurrency-correctness test.

Asserts that transcribing N clips sequentially yields the SAME texts as
transcribing them concurrently via asyncio.gather.

Also contains fast unit tests that need no GPU.
"""
import asyncio
import pytest

pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Fast unit tests — no GPU, no model load
# ---------------------------------------------------------------------------

def test_scheduler_rejects_zero_concurrency():
    """TwoPhaseScheduler(max_concurrency=0) must raise ValueError."""
    from parakeet_vllm.scheduling.two_phase_scheduler import TwoPhaseScheduler

    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        TwoPhaseScheduler(max_concurrency=0)


def test_scheduler_rejects_negative_concurrency():
    """TwoPhaseScheduler(max_concurrency=-1) must raise ValueError."""
    from parakeet_vllm.scheduling.two_phase_scheduler import TwoPhaseScheduler

    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        TwoPhaseScheduler(max_concurrency=-1)


@pytest.mark.gpu
def test_concurrent_requests_are_correct():
    from datasets import load_dataset, Audio
    from parakeet_vllm.engine.asr_engine import ASREngine

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    samples = [ds["audio"][i]["array"].astype("float32") for i in range(4)]
    eng = ASREngine()

    async def run():
        single = [
            (await eng.transcribe(s, f"s{i}")).text for i, s in enumerate(samples)
        ]
        conc = await asyncio.gather(
            *[eng.transcribe(s, f"c{i}") for i, s in enumerate(samples)]
        )
        return single, [c.text for c in conc]

    single, conc = asyncio.run(run())
    assert single == conc, (
        f"Concurrent results differ from sequential:\n"
        f"  sequential: {single}\n"
        f"  concurrent: {conc}"
    )

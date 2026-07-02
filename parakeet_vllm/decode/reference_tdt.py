from __future__ import annotations

from typing import AsyncIterator

import torch

from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderModelOutput

from .backend import DecodeBackend, TokenDelta
from .tdt_step import split_joint


class ReferenceTDTBackend(DecodeBackend):
    """Greedy TDT decode driving ``ParakeetForTDT.forward`` step-by-step.

    Ports transformers' Parakeet TDT greedy loop (see
    ``docs/superpowers/findings/2026-06-30-tdt-decode-loop.md``) so it can be
    fed precomputed encoder frames and run outside ``model.generate()``.

    The upstream generate() reuses the base ``GenerationMixin._sample`` greedy
    loop; ``ParakeetTDTGenerationMixin`` only hooks encoder-frame slicing,
    duration-based frame advance, the decoder cache, and encoder-exhaustion
    stopping. This backend reimplements exactly that machinery for batch size 1
    and reproduces ``oracle.sequences[0]`` token-for-token (leading
    ``decoder_start_token_id`` and per-step blanks included).
    """

    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.blank_id = int(self.cfg.blank_token_id)
        self.vocab_size = int(self.cfg.vocab_size)
        self.durations = list(self.cfg.durations)
        self.max_symbols = int(self.cfg.max_symbols_per_step)

        # Selection / stopping parameters mirror model.generate defaults.
        gen_cfg = getattr(model, "generation_config", None)
        start = getattr(gen_cfg, "decoder_start_token_id", None) if gen_cfg else None
        if start is None:
            start = getattr(self.cfg, "decoder_start_token_id", None)
        self.decoder_start_token_id = int(start) if start is not None else self.blank_id
        eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
        if isinstance(eos, (list, tuple)):
            self.eos_token_ids = {int(e) for e in eos}
        elif eos is not None:
            self.eos_token_ids = {int(eos)}
        else:
            self.eos_token_ids = set()

    def _split_joint(self, logits: torch.Tensor) -> tuple[int, int]:
        """Split the joint output into (token_id, frame_advance).

        ``logits`` is ``(1, 1, vocab_size + len(durations))`` from forward.
        Token id is the argmax over the vocab slice; because generate suppresses
        the duration slots (generation_config.suppress_tokens), this equals the
        full-width argmax it uses to build ``sequences``. The duration is the
        argmax index over the duration slice mapped through ``config.durations``.
        Blank predictions with duration 0 are forced to advance 1 frame.
        """
        return split_joint(
            logits[0, -1],
            vocab_size=self.vocab_size,
            blank_id=self.blank_id,
            durations=self.durations,
        )

    async def decode_stream(
        self, request_id, encoder_frames, encoder_lengths
    ) -> AsyncIterator[TokenDelta]:
        device = encoder_frames.device
        T = int(encoder_lengths[0].item())

        frame_ptr = 0
        prev_token = torch.tensor([[self.decoder_start_token_id]], device=device)
        decoder_cache = None
        produced = 0
        max_produced = self.max_symbols * T  # generate's max_length safety cap

        # Per-frame non-advancing emission counter (standard RNN-T
        # max_symbols_per_step guard).  Reset whenever the frame pointer
        # advances; if it reaches max_symbols on the same frame, force a
        # frame advance so the loop can never spin on one frame forever.
        symbols_on_frame = 0

        # sequences[0] is the prepended decoder_start_token_id (duration 0).
        yield TokenDelta(
            token_ids=[self.decoder_start_token_id], durations=[0], finished=False
        )

        try:
            with torch.inference_mode():
                # forward's joint consumes the projected pooler_output (decoder_hidden_size),
                # not the raw encoder last_hidden_state. get_audio_features applies this same
                # projection internally; the test hands us raw frames so we project here.
                pooler_full = self.model.encoder_projector(encoder_frames)  # (1, T, D)

                while True:
                    frame = min(frame_ptr, T - 1)
                    encoder_outputs = ParakeetEncoderModelOutput(
                        pooler_output=pooler_full[:, frame : frame + 1, :]
                    )
                    out = self.model(
                        decoder_input_ids=prev_token,
                        decoder_cache=decoder_cache,
                        use_decoder_cache=True,
                        encoder_outputs=encoder_outputs,
                    )
                    decoder_cache = out.decoder_cache

                    token_id, duration = self._split_joint(out.logits)

                    # Per-frame symbol cap: a non-blank token with duration 0
                    # does not advance the frame pointer.  After max_symbols such
                    # emissions on one frame, force a frame advance to prevent an
                    # infinite loop (mirrors ParakeetTDTGenerationMixin behaviour).
                    if token_id != self.blank_id and duration == 0:
                        symbols_on_frame += 1
                        if symbols_on_frame >= self.max_symbols:
                            duration = 1  # force advance past this frame
                            symbols_on_frame = 0
                    else:
                        # Any blank or advancing emission resets the per-frame counter.
                        symbols_on_frame = 0

                    # Emit every token (blanks included), matching generate's sequences.
                    yield TokenDelta(
                        token_ids=[token_id], durations=[duration], finished=False
                    )
                    produced += 1
                    prev_token = torch.tensor([[token_id]], device=device)
                    frame_ptr += duration

                    # Stopping mirrors generate: encoder exhaustion, eos, max_length cap.
                    # (token appended before the stop check, so it stays in the output.)
                    if frame_ptr >= T:
                        break
                    if token_id in self.eos_token_ids:
                        break
                    if 1 + produced >= max_produced:
                        break
        finally:
            # Release the KV cache on any exit (normal finish, break, or
            # GeneratorExit / cancellation from an abandoned async-for loop).
            decoder_cache = None

        yield TokenDelta(token_ids=[], durations=[], finished=True)

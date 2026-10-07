"""Generate an audio transition between two songs using the trained BERT model.

The BERT model in transformer_models/bert was trained on a masked-token-infilling
task: given a window of INTERVAL_LENGTH*SAMPLE_RATE audio tokens with a
MASK_LENGTH*SAMPLE_RATE-token gap in the middle (filled with tokens shuffled from
the surrounding context, see data/data_loader.py), predict the original tokens for
that gap. There is no "mix two songs" task anywhere in this codebase or model, so
this script builds the closest legitimate equivalent: it puts the tail of song A
and the head of song B on either side of that gap and asks the model to fill it
in, i.e. to generate a short bridge between the two songs rather than just cutting
from one to the other.

Usage (run from the repo root):
    python validation/generate_transition.py song_a.wav song_b.wav out.wav
"""
import argparse
import os
import sys

import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer_models.bert import config
from transformer_models.bert.model.model import BERT_model


def load_wav(path):
    """Read audio as a [channels, samples] float32 tensor. We use soundfile instead of
    torchaudio.load because recent torchaudio versions route audio I/O through
    torchcodec, which in this environment pulls in a CUDA-only build that fails to
    import on a CPU machine."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
    return torch.from_numpy(data.T), sr


def save_wav(path, wav, sample_rate):
    """wav: [channels, samples] float32 tensor."""
    sf.write(path, wav.numpy().T, sample_rate)


def encode_audio(path, encodec_model, device):
    wav, sr = load_wav(path)
    wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
    wav = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1)  # [1, n_q, T]
    return codes[:, : config.AUDIO_CHANNELS, :]


def decode_tokens(tokens, encodec_model, device):
    # tokens: [T, channels] -> [1, channels, T]
    codes = tokens.transpose(0, 1).unsqueeze(0).long().to(device)
    with torch.no_grad():
        audio = encodec_model.decode([(codes, None)])
    return audio.squeeze(0).cpu()  # [channels, samples]


def crossfade_concat(a, b, fade_samples):
    """Concatenate two [channels, samples] waveforms with a linear crossfade,
    to soften the seam between independently-decoded Encodec chunks."""
    fade_samples = min(fade_samples, a.shape[-1], b.shape[-1])
    if fade_samples <= 0:
        return torch.cat([a, b], dim=-1)
    fade_out = torch.linspace(1.0, 0.0, fade_samples)
    fade_in = torch.linspace(0.0, 1.0, fade_samples)
    a_body, a_tail = a[..., :-fade_samples], a[..., -fade_samples:]
    b_head, b_body = b[..., :fade_samples], b[..., fade_samples:]
    mixed = a_tail * fade_out + b_head * fade_in
    return torch.cat([a_body, mixed, b_body], dim=-1)


def load_models(checkpoint_path, device):
    """Load the Encodec codec and the trained BERT model once, so callers that
    generate multiple transitions (e.g. a GUI) don't reload them every time."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train the model first (transformer_models/bert/train.py) "
            "or point at an existing one."
        )

    encodec_model = EncodecModel.encodec_model_48khz().to(device)
    encodec_model.set_target_bandwidth(6.0)

    bert_model = BERT_model(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dropout=config.DROPOUT,
    ).to(device)
    bert_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    bert_model.eval()

    return encodec_model, bert_model


def generate_transition(song_a, song_b, output, encodec_model, bert_model, device, crossfade_ms=50.0):
    """Generate a transition between song_a and song_b using already-loaded models
    and write the result to `output`. Raises ValueError if a song is too short."""
    interval_length = config.INTERVAL_LENGTH * config.SAMPLE_RATE  # total window, in tokens
    mask_length = config.MASK_LENGTH * config.SAMPLE_RATE  # gap to fill, in tokens
    context_half = (interval_length - mask_length) // 2  # tokens of real context on each side
    samples_per_token = encodec_model.sample_rate / encodec_model.frame_rate

    tokens_a = encode_audio(song_a, encodec_model, device).squeeze(0).transpose(0, 1)  # [T_a, channels]
    tokens_b = encode_audio(song_b, encodec_model, device).squeeze(0).transpose(0, 1)  # [T_b, channels]

    if tokens_a.shape[0] < context_half:
        raise ValueError(f"{song_a} is shorter than the {context_half}-token context window the model needs.")
    if tokens_b.shape[0] < context_half:
        raise ValueError(f"{song_b} is shorter than the {context_half}-token context window the model needs.")

    tail_a = tokens_a[-context_half:]
    head_b = tokens_b[:context_half]

    # Seed the gap with tokens shuffled from the surrounding context -- this matches
    # the "shuffle" noise strategy the model was trained to denoise (data/data_loader.py),
    # rather than a blank/zero gap the model has never seen at train time.
    context = torch.cat([tail_a, head_b], dim=0)
    shuffle_idx = torch.randperm(context.shape[0])[:mask_length]
    noise = context[shuffle_idx]

    window = torch.cat([tail_a, noise, head_b], dim=0)  # [interval_length, channels]
    assert window.shape[0] == interval_length

    input_ids = window.unsqueeze(0).long().to(device)  # [1, interval_length, channels]
    with torch.no_grad():
        logits, _ = bert_model(input_ids, input_ids, None)
    logits = logits.reshape(1, interval_length, config.AUDIO_CHANNELS, config.VOCAB_SIZE)
    predicted = torch.argmax(logits, dim=-1).squeeze(0).cpu()  # [interval_length, channels]

    mask_start = context_half
    mask_end = mask_start + mask_length
    bridge = window.clone()
    bridge[mask_start:mask_end] = predicted[mask_start:mask_end]

    audio_a, sr_a = load_wav(song_a)
    audio_a = convert_audio(audio_a, sr_a, encodec_model.sample_rate, encodec_model.channels)
    cut_a = audio_a.shape[-1] - int(context_half * samples_per_token)
    head_audio_a = audio_a[:, : max(cut_a, 0)]

    audio_b, sr_b = load_wav(song_b)
    audio_b = convert_audio(audio_b, sr_b, encodec_model.sample_rate, encodec_model.channels)
    cut_b = int(context_half * samples_per_token)
    tail_audio_b = audio_b[:, cut_b:]

    bridge_audio = decode_tokens(bridge, encodec_model, device)

    fade_samples = int(crossfade_ms / 1000 * encodec_model.sample_rate)
    full = crossfade_concat(head_audio_a, bridge_audio, fade_samples)
    full = crossfade_concat(full, tail_audio_b, fade_samples)

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    save_wav(output, full, encodec_model.sample_rate)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("song_a", help="Path to the first song (plays first)")
    parser.add_argument("song_b", help="Path to the second song (plays second)")
    parser.add_argument("output", help="Path to write the mixed .wav to")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH, help="BERT checkpoint to use")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--crossfade-ms", type=float, default=50.0, help="Crossfade at each splice point, in ms")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading Encodec and BERT model...")
    encodec_model, bert_model = load_models(args.checkpoint, device)

    print(f"Encoding {args.song_a} and {args.song_b}, generating bridge, decoding...")
    generate_transition(args.song_a, args.song_b, args.output, encodec_model, bert_model, device, args.crossfade_ms)
    print(f"Wrote transition to {args.output}")


if __name__ == "__main__":
    main()

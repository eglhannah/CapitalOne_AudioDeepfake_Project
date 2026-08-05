"""
dash_aasist_app.py
===================
Temporary Dash dashboard for AASIST v3 audio deepfake detection
with explanation (embedding SHAP, occlusion, spectrogram SHAP).

Run with:
    python dash_aasist_app.py
Then open http://127.0.0.1:8050 in your browser.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import torch
import torchaudio
import soundfile as sf
import subprocess
import struct
import imageio_ffmpeg

sys.path.append(str(Path(__file__).resolve().parents[1]))
from aasist.simple_aasist import load_aasist_v3, predict

from interpret_aasist_shap import (
    TARGET_LENGTH, SR, MEL_N_MELS, MEL_N_FFT, MEL_HOP_LENGTH,
    fix_length,
    compute_mel_spectrograms,
    SpectrogramSurrogate,
    train_surrogate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AASIST v3 model...")
MODEL = load_aasist_v3(device=str(DEVICE))
MODEL.eval()
print(f"Model loaded on {DEVICE}")

EMPTY_FIG = go.Figure()
EMPTY_FIG.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    annotations=[dict(text="", showarrow=False, font=dict(size=16, color="#888"))],
    margin=dict(l=20, r=20, t=20, b=20), height=200,
)


def _blank_fig(msg=""):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=msg, showarrow=False, font=dict(size=14, color="#888"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        margin=dict(l=20, r=20, t=20, b=20), height=200,
    )
    return fig


def _error_fig(msg):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(30,0,0,0.3)",
        plot_bgcolor="rgba(30,0,0,0.3)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=f"Error: {msg}", showarrow=False,
                          font=dict(size=13, color="#e74c3c"),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        margin=dict(l=20, r=20, t=20, b=20), height=200,
    )
    return fig


_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

AUDIO_EXTENSIONS = {".flac", ".wav"}
COMPRESSED_EXTENSIONS = {".mp3", ".aac", ".m4a", ".ogg", ".wma", ".opus"}


def _ffmpeg_decode(tmp_path: str) -> np.ndarray:
    """Use ffmpeg to decode any audio format to 16kHz mono float32 PCM via stdout."""
    cmd = [
        _FFMPEG_EXE,
        "-i", tmp_path,
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode('utf-8', errors='replace')[:500]}")

    wav_bytes = result.stdout
    if len(wav_bytes) < 44:
        raise RuntimeError("ffmpeg produced empty output")

    header = wav_bytes[:44]
    data_offset = 44
    num_channels = struct.unpack_from("<H", header, 22)[0]
    bits_per_sample = struct.unpack_from("<H", header, 34)[0]

    if bits_per_sample == 16:
        samples = np.frombuffer(wav_bytes[data_offset:], dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(wav_bytes[data_offset:], dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported bit depth: {bits_per_sample}")

    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)

    return samples


def load_any_audio(file_bytes: bytes, filename: str) -> torch.Tensor:
    suffix = Path(filename).suffix.lower()
    tmp_path = os.path.join(tempfile.gettempdir(), f"aasist_upload{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    try:
        if suffix in COMPRESSED_EXTENSIONS:
            samples = _ffmpeg_decode(tmp_path)
        else:
            data, sr = sf.read(tmp_path)
            samples = data.astype(np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if sr != 16000:
                wav_tensor = torch.from_numpy(samples).unsqueeze(0)
                wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
                samples = wav_tensor.squeeze(0).numpy()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return fix_length(torch.from_numpy(samples).float())


def wav_to_wav_bytes(wav: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.numpy(), 16000, format="WAV")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def make_spectrogram_figure(wav: torch.Tensor) -> go.Figure:
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS,
    )
    spec = mel_spec(wav.unsqueeze(0))
    spec_db = torch.log(spec.clamp(min=1e-9)).squeeze(0).numpy()
    duration = wav.shape[0] / SR
    time_axis = np.linspace(0, duration, spec_db.shape[1])
    freq_axis = np.arange(MEL_N_MELS)

    fig = go.Figure(data=go.Heatmap(
        z=spec_db, x=time_axis, y=freq_axis,
        colorscale="Viridis", colorbar=dict(title="Log-Mel"),
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Mel Spectrogram",
        xaxis_title="Time (s)", yaxis_title="Mel frequency bin",
        height=350, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def make_embedding_shap_bar(shap_vals: np.ndarray) -> go.Figure:
    top_n = 20
    top_idx = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
    vals = shap_vals[top_idx]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=[f"dim {i}" for i in top_idx],
        orientation="h", marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Top-20 Embedding Dimensions (red=spoof, blue=bonafide)",
        xaxis_title="SHAP value", height=450,
        margin=dict(l=40, r=20, t=40, b=40), yaxis=dict(autorange="reversed"),
    )
    return fig


def make_occlusion_figure(occl: np.ndarray) -> go.Figure:
    n_windows = occl.shape[0]
    window_len = TARGET_LENGTH // n_windows
    time_axis = np.arange(n_windows) * window_len / SR
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in occl]

    fig = go.Figure(go.Bar(
        x=time_axis, y=occl, marker_color=colors,
        width=window_len / SR * 0.9,
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Occlusion Sensitivity (red=removing hurts spoof, blue=removing hurts bonafide)",
        xaxis_title="Time (s)", yaxis_title="Delta P(spoof)",
        height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    return fig


def make_spec_shap_figure(shap_vals: np.ndarray) -> go.Figure:
    duration = shap_vals.shape[1] * MEL_HOP_LENGTH / SR
    time_axis = np.linspace(0, duration, shap_vals.shape[1])
    freq_axis = np.arange(shap_vals.shape[0])
    vmax = np.max(np.abs(shap_vals))

    fig = go.Figure(data=go.Heatmap(
        z=shap_vals, x=time_axis, y=freq_axis,
        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="SHAP"),
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Spectrogram SHAP (red=spoof, blue=bonafide)",
        xaxis_title="Time (s)", yaxis_title="Mel frequency bin",
        height=350, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app layout
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

STATUS_DIV = html.Div(
    id="status-bar",
    children=dbc.Alert(
        "Loading model...",
        color="info", className="d-flex align-items-center mb-3",
    ),
)

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("AASIST v3 - Audio Deepfake Explanation Dashboard"),
                     width=12), className="my-3"),

    STATUS_DIV,

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-audio",
                children=dbc.Button("Upload Audio File", color="primary",
                                    id="upload-btn", className="mb-2"),
                accept=".flac,.wav,.mp3,.aac,.m4a,.ogg,.wma,.opus",
            ),
            html.Div(id="upload-filename", className="text-muted mb-2"),
            html.Div(id="audio-player-div"),
        ], width=4),

        dbc.Col([
            html.Div(id="prediction-output"),
        ], width=8),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-spectrogram",
                type="circle",
                children=dcc.Graph(id="spectrogram-graph", figure=_blank_fig()),
            ),
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-embedding",
                type="circle",
                children=dcc.Graph(id="embedding-shap-graph", figure=_blank_fig()),
            ),
        ], width=6),
        dbc.Col([
            dcc.Loading(
                id="loading-occlusion",
                type="circle",
                children=dcc.Graph(id="occlusion-graph", figure=_blank_fig()),
            ),
        ], width=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-specshap",
                type="circle",
                children=dcc.Graph(id="spec-shap-graph", figure=_blank_fig()),
            ),
        ], width=12),
    ], className="mb-3"),

    html.Div(id="hidden-state", style={"display": "none"}),
    dcc.Interval(id="init-interval", interval=500, n_intervals=0, max_intervals=1),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("status-bar", "children"),
    Output("upload-btn", "disabled"),
    Input("init-interval", "n_intervals"),
    prevent_initial_call=True,
)
def show_model_ready(_):
    return (
        dbc.Alert("Model loaded. Ready for audio file.",
                  color="success", className="d-flex align-items-center mb-3"),
        False,
    )


@callback(
    Output("status-bar", "children", allow_duplicate=True),
    Output("hidden-state", "children"),
    Output("upload-filename", "children"),
    Output("audio-player-div", "children"),
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def process_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    status_analyzing = dbc.Alert(
        dbc.Spinner(size="sm", color="light"),
        " Analyzing audio and extracting explanations...",
        color="warning", className="d-flex align-items-center mb-3",
    )

    header, data = contents.split(",")
    file_bytes = base64.b64decode(data)
    wav = load_any_audio(file_bytes, filename)

    wav_bytes = wav_to_wav_bytes(wav)
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    audio_player = html.Audio(
        src=f"data:audio/wav;base64,{wav_b64}",
        controls=True,
        style={"width": "100%", "margin-top": "10px"},
    )

    return status_analyzing, wav.numpy().tolist(), filename, audio_player


@callback(
    Output("status-bar", "children", allow_duplicate=True),
    Output("prediction-output", "children"),
    Output("spectrogram-graph", "figure"),
    Output("embedding-shap-graph", "figure"),
    Output("occlusion-graph", "figure"),
    Output("spec-shap-graph", "figure"),
    Input("hidden-state", "children"),
    prevent_initial_call=True,
)
def run_explanation(wav_list):
    if not wav_list:
        raise dash.exceptions.PreventUpdate

    wav = torch.tensor(wav_list, dtype=torch.float32).to(DEVICE)

    # --- Prediction (must succeed) ---
    with torch.no_grad():
        out = predict(MODEL, wav.unsqueeze(0))
    prob = out["spoof_prob"].item()
    pred = "SPOOF" if prob >= 0.5 else "BONAFIDE"
    label_class = "danger" if prob >= 0.5 else "success"

    pred_card = dbc.Card([
        dbc.CardHeader(html.H4("Prediction")),
        dbc.CardBody([
            html.H3(pred, className=f"text-{label_class}"),
            html.P(f"Spoof probability: {prob:.4f}"),
            html.P(f"Confidence: {max(prob, 1 - prob):.2%}"),
        ]),
    ], color=label_class, outline=True)

    wav_cpu = wav.cpu()

    # --- Spectrogram ---
    try:
        spec_fig = make_spectrogram_figure(wav_cpu)
    except Exception as e:
        spec_fig = _error_fig(str(e))

    # --- Embedding SHAP ---
    try:
        from interpret_aasist_shap import AASISTEmbeddingHead
        import shap as shap_lib
        shap_head = AASISTEmbeddingHead(MODEL).to(DEVICE).eval()
        with torch.no_grad():
            emb = out["embedding"].squeeze(0)
        bg_emb = emb.unsqueeze(0).repeat(20, 1)
        noise = torch.randn_like(bg_emb) * 0.05 * bg_emb.abs().clamp(min=1e-3)
        bg_emb = bg_emb + noise
        explainer = shap_lib.DeepExplainer(shap_head, bg_emb.to(DEVICE))
        sv = explainer.shap_values(emb.unsqueeze(0).to(DEVICE), check_additivity=False)
        sv = np.asarray(sv).flatten()
        if np.all(sv == 0) or np.any(np.isnan(sv)):
            emb_fig = _error_fig("SHAP returned all-zero or NaN values")
        else:
            emb_fig = make_embedding_shap_bar(sv)
    except Exception as e:
        emb_fig = _error_fig(str(e))

    # --- Occlusion sensitivity ---
    try:
        n_windows = 100
        window_len = TARGET_LENGTH // n_windows
        occl = np.zeros(n_windows)
        for w in range(n_windows):
            occluded = wav_cpu.clone()
            occluded[w * window_len:(w + 1) * window_len] = 0.0
            with torch.no_grad():
                occ_out = predict(MODEL, occluded.unsqueeze(0).to(DEVICE))
            occl[w] = prob - occ_out["spoof_prob"].item()
        occl_fig = make_occlusion_figure(occl)
    except Exception as e:
        occl_fig = _error_fig(str(e))

    # --- Spectrogram SHAP via surrogate ---
    try:
        import shap as shap_lib
        specs = compute_mel_spectrograms(wav_cpu.unsqueeze(0))
        spec_flat = specs.reshape(1, -1)
        n_features = spec_flat.shape[1]

        bg_spec_data = spec_flat.repeat(20, 1)
        noise = torch.randn_like(bg_spec_data) * 0.05 * bg_spec_data.abs().clamp(min=1e-3)
        bg_spec_data = bg_spec_data + noise
        bg_target = bg_spec_data  # surrogate will be trained on the actual sample, background is for SHAP

        targets = torch.tensor([[prob]], dtype=torch.float32)
        surrogate = SpectrogramSurrogate(n_features)
        train_surrogate(surrogate, spec_flat, targets, epochs=200, device=str(DEVICE))
        surrogate.eval()

        bg_spec_dev = bg_spec_data.to(DEVICE)
        spec_explainer = shap_lib.DeepExplainer(surrogate, bg_spec_dev)
        spec_sv = spec_explainer.shap_values(spec_flat.to(DEVICE), check_additivity=False)
        spec_sv = np.asarray(spec_sv)
        if spec_sv.ndim == 3:
            spec_sv = spec_sv.squeeze(-1)
        if spec_sv.ndim == 1:
            spec_sv = spec_sv.reshape(MEL_N_MELS, -1)
        elif spec_sv.ndim == 2:
            spec_sv = spec_sv.reshape(MEL_N_MELS, -1)
        if np.all(spec_sv == 0) or np.any(np.isnan(spec_sv)):
            spec_shap_fig = _error_fig("SHAP returned all-zero or NaN values")
        else:
            spec_shap_fig = make_spec_shap_figure(spec_sv)
    except Exception as e:
        spec_shap_fig = _error_fig(str(e))

    status_done = dbc.Alert(
        f"Analysis complete - Prediction: {pred} (P(spoof)={prob:.4f})",
        color="success" if prob < 0.5 else "danger",
        className="d-flex align-items-center mb-3",
    )

    return status_done, pred_card, spec_fig, emb_fig, occl_fig, spec_shap_fig


if __name__ == "__main__":
    print("Starting Dash app at http://127.0.0.1:8050")
    app.run(debug=False, port=8050)

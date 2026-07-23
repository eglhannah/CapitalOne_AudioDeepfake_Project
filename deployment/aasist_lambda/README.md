# AASIST v3 Lambda deployment

This directory contains the reproducible deployment path for the AASIST v3
CodecAugment checkpoint. It includes artifact provenance, audio decoding,
waveform scoring, a Lambda container, and a local browser demo client. AWS
resource creation is intentionally kept for a later deployment phase.

## Pinned inputs

- Model: `arnavjain321/aasist-v3-codecaugment`
- Active checkpoint: `artifacts/aasist_v3_best.pt`
- Active checkpoint SHA-256:
  `36e27b4b2032c0a7448f7c9dab2db89efd607013b8596da2b7be8419814b83d0`
- Implementation: `clovaai/aasist` at Git commit
  `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
- The source implementation is distributed under the included MIT license.

Model checkpoints are deliberately ignored by Git. The active checkpoint path,
expected size, and SHA-256 digest are recorded in `artifact-manifest.json`.

## Phase 1 verification

From this directory:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-local.txt
.venv/bin/python scripts/fetch_model_artifacts.py
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/smoke_test.py
.venv/bin/python -m unittest discover -s tests -v
```

The smoke test uses deterministic synthetic noise only to prove that the model
loads strictly and executes repeatably. Known bonafide and spoof examples will
be added as golden behavioral fixtures without committing dataset audio.

## Inference contract

- Input waveform: mono float32 PCM at 16 kHz
- Native window: 64,600 samples (4.0375 seconds)
- Planned long-audio stride: 32,000 samples (2 seconds)
- Class index 0: bonafide
- Class index 1: spoof
- Initial, uncalibrated decision threshold: 0.5

The API should always return the continuous spoof score. The `0.5` label is a
demo default and must not be represented as a calibrated real-world risk
boundary.

## Phase 2 waveform scorer

`AASISTScorer` accepts an already-decoded, mono float32 NumPy waveform at
16 kHz. It repeat-pads short clips and scores longer clips with 64,600-sample
windows at a 32,000-sample stride. A final tail-aligned window guarantees that
no decoded audio is omitted. Window spoof probabilities are averaged.

Score a supported audio file or predecoded waveform:

```bash
.venv/bin/python -m aasist_inference local_samples/example.flac
.venv/bin/python -m aasist_inference waveform.npy --sample-rate 16000
```

Run the local CPU benchmark:

```bash
.venv/bin/python scripts/benchmark_scorer.py
```

Place private ASVspoof examples in `local_samples/`. The entire directory is
ignored by Git.

## Phase 3 audio decoding

Audio uploads are identified by container signature rather than extension and
decoded entirely through FFmpeg stdin/stdout. Nothing is written to disk. The
decoder accepts WAV, FLAC, MP3, M4A/AAC, OGG, and WebM, converts stereo to mono,
and resamples to 16 kHz float32 PCM. Encoded uploads are limited to 4 MiB and
decoded audio to 30 seconds. FFmpeg output is itself duration-bounded to prevent
small compressed files from expanding without limit.

For local development, `imageio-ffmpeg` supplies a pinned executable when no
system FFmpeg exists. The Lambda image will use a system FFmpeg binary instead.

Score every private sample without writing results:

```bash
.venv/bin/python scripts/score_local_samples.py
.venv/bin/python scripts/validate_format_consistency.py
```

## Phase 4 Lambda container

The container targets one `linux/amd64` platform and uses a digest-pinned
Python 3.12 Debian base, CPU-only PyTorch, a pinned static FFmpeg executable,
AWS Lambda Runtime Interface Client 4.0.0, and Runtime Interface Emulator
v1.35. Model weights are included in the image and verified during the build;
runtime inference requires no network access.

Build and start it with Lambda-like constraints:

```bash
make container-build
make container-run
```

The container runs with one CPU, 2 GiB memory, a read-only root filesystem,
and a 64 MiB `/tmp`. In another terminal:

```bash
make container-test
make container-benchmark
make container-stop
```

`container-test` compares native and container scores for both private samples
and checks the 405, 413, 415, and 422 responses. The private samples are mounted
only for offline verification and are excluded from the build context. Local
measurements are recorded in `results/phase4-container-report.json`; because
the development host is Apple ARM, x86 timings include substantial emulation
overhead and are not AWS performance estimates.

## Phase 5 presentation demo client

Phase 5 adds a small browser client and lightweight demo access controls while
still avoiding AWS resources. The Lambda handler now supports:

- CORS headers for browser calls.
- `OPTIONS` preflight requests.
- Optional passcode protection through `DEMO_PASSPHRASE`.
- Configurable browser origin through `DEMO_ALLOWED_ORIGIN`.

If `DEMO_PASSPHRASE` is unset or empty, passcode checks are disabled. If it is
set, clients must send the same value in the `x-demo-passcode` request header.
This is presentation-grade gating, not a replacement for production
authentication.

Run the full local demo:

```bash
make container-build
make container-run DEMO_PASSPHRASE=demo-pass
make demo-local
```

Then open <http://127.0.0.1:8765>, enter `demo-pass`, upload a supported audio
file, and click **Analyze audio**.

The local browser client posts raw audio to `/infer`. The development server in
`scripts/serve_demo_client.py` converts that upload into the Lambda Runtime
Interface Emulator invocation shape expected by the local container. In AWS, the
same static page can point directly at a Lambda Function URL because AWS will
perform that event translation.

To test without a passcode:

```bash
make container-run
make demo-local
```

Supported demo upload containers remain WAV, FLAC, MP3, M4A/AAC, OGG, and WebM.
Uploads are still limited to 4 MiB, decoded audio is still capped at 30 seconds,
and audio is discarded immediately after inference.

The demo client also provides local playback for the selected upload before and
after inference. Playback first uses the original browser-selected file. If the
browser cannot preview that container directly, the page attempts a local Web
Audio decode and creates a temporary WAV preview in the browser. This does not
store audio in S3, send audio to Lambda for playback, or change the original
file sent for inference.

The static client can also record up to 15 seconds from the user's microphone.
The recording is kept in the browser as a temporary WebM/Opus-style Blob when
supported by Chrome, becomes the active selected input, and is handled by the
same preview and inference path as an uploaded file. Microphone access works on
local trusted origins such as `http://127.0.0.1`; the hosted S3 website will
need HTTPS, such as CloudFront in front of S3, before recording is available
from AWS.

## Phase 6 AWS deployment

Phase 6 deploys the v3 container and HTTPS browser demo to AWS with a small,
teardown-friendly footprint. Resources are split into backend and frontend
layers so ECR/Lambda can be torn down between rehearsals while the slower
CloudFront HTTPS frontend remains available.

- Backend: ECR private repository, Lambda container function, Function URL,
  IAM role, and CloudWatch log retention.
- Frontend: S3 static bucket and CloudFront HTTPS distribution.
- Lambda Function URL uses `AuthType=NONE`, Function URL CORS, and app-level
  passcode.
- CloudWatch log retention set to 7 days by default.
- Optional Lambda reserved concurrency through `LAMBDA_RESERVED_CONCURRENCY`.
- Lambda defaults to 2048 MB memory and a 90-second timeout.
- CloudFront uses `PriceClass_100` by default and the generated
  `*.cloudfront.net` HTTPS URL.

The scripts refuse to run unless `EXPECTED_AWS_ACCOUNT_ID` matches
`aws sts get-caller-identity`.

Deploy from nothing to the full HTTPS demo:

```bash
cd /Users/chasecha/Desktop/CapitalOne_AudioDeepfake_Project/deployment/aasist_lambda
EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
DEMO_PASSPHRASE='choose-a-temporary-demo-passphrase' \
./aws/deploy-backend.sh

EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
./aws/deploy-frontend.sh
```

Override Lambda size if needed:

```bash
LAMBDA_MEMORY_MB=2048
LAMBDA_TIMEOUT_SECONDS=90
```

Some small AWS accounts cannot reserve concurrency without dropping the
account's unreserved pool below AWS's minimum. In that case, leave
`LAMBDA_RESERVED_CONCURRENCY` unset. If the account allows it and you want the
extra cost guardrail, add for example:

```bash
LAMBDA_RESERVED_CONCURRENCY=1
```

The deploy script writes generated resource details to
`aws/backend-info.json`, `aws/frontend-info.json`, and a compatibility
`aws/deployment-info.json`. These generated files are ignored by Git.

Smoke test the deployed Function URL with the known bonafide private sample:

```bash
DEMO_PASSPHRASE='choose-a-temporary-demo-passphrase' \
./aws/smoke-test.sh --expect bonafide
```

Open the printed CloudFront URL in Chrome, enter the same passphrase, upload or
record audio, and run inference. The hosted page loads its Function URL from the
generated `config.js` object uploaded during frontend deployment. Microphone
recording requires this HTTPS CloudFront URL; the S3 bucket URL is not the demo
URL.

If only the static browser client changes, update the S3 page without rebuilding
the Lambda image or changing CloudFront:

```bash
./aws/update-static-client.sh
```

To save backend/ECR costs after rehearsal while keeping HTTPS static hosting
alive:

```bash
EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
CONFIRM_TEARDOWN=aasist-audio-deepfake-demo \
./aws/teardown-backend.sh
```

To restore the backend later and refresh the CloudFront-hosted client with the
new Function URL:

```bash
EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
DEMO_PASSPHRASE='choose-a-temporary-demo-passphrase' \
./aws/deploy-backend.sh

EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
./aws/deploy-frontend.sh
```

The first `deploy-frontend.sh` run creates CloudFront and can take many minutes.
Later runs reuse the existing distribution, upload `index.html`/`config.js`, and
create a small invalidation for `/index.html` and `/config.js`.

After the final presentation, tear everything down:

```bash
EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
CONFIRM_TEARDOWN=aasist-audio-deepfake-demo \
./aws/teardown-backend.sh

EXPECTED_AWS_ACCOUNT_ID=857622871695 \
AWS_REGION=us-east-1 \
CONFIRM_TEARDOWN=aasist-audio-deepfake-demo \
./aws/teardown-frontend.sh
```

`deploy.sh` and `teardown.sh` remain compatibility wrappers for
`deploy-backend.sh` and `teardown-backend.sh`.

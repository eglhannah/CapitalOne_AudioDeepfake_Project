# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS dependency-builder
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir --target /opt/python \
      --requirement /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir --target /opt/python \
      --index-url https://download.pytorch.org/whl/cpu \
      --no-deps \
      'torch==2.7.1+cpu'
RUN python -m pip install --no-cache-dir --target /opt/ffmpeg-package 'imageio-ffmpeg==0.6.0'

ARG RIE_VERSION=v1.35
ARG RIE_SHA256=3a3494331c212c14366c1073fdc4f0078f58a927da64479413879b0d84e21f20
RUN PYTHONPATH=/opt/ffmpeg-package python - <<EOF
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
import imageio_ffmpeg

output = Path("/opt/bin")
output.mkdir(parents=True, exist_ok=True)
shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), output / "ffmpeg")
urllib.request.urlretrieve(
    "https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/${RIE_VERSION}/aws-lambda-rie-x86_64",
    output / "aws-lambda-rie",
)
digest = hashlib.sha256((output / "aws-lambda-rie").read_bytes()).hexdigest()
if digest != "${RIE_SHA256}":
    raise SystemExit(f"RIE checksum mismatch: {digest}")
os.chmod(output / "ffmpeg", 0o555)
os.chmod(output / "aws-lambda-rie", 0o555)
EOF

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS artifact-verifier
WORKDIR /source
COPY artifact-manifest.json ./
COPY artifacts/ ./artifacts/
COPY vendor/ ./vendor/
COPY scripts/verify_artifacts.py ./scripts/verify_artifacts.py
RUN python scripts/verify_artifacts.py

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS runtime

ARG MODEL_REVISION=sha256:36e27b4b2032c0a7448f7c9dab2db89efd607013b8596da2b7be8419814b83d0
ARG AASIST_REVISION=a04c9863f63d44471dde8a6abcb3b082b07cd1d1

LABEL org.opencontainers.image.title="AASIST v3 CodecAugment Lambda inference" \
      org.opencontainers.image.description="CPU audio deepfake inference for AWS Lambda" \
      org.opencontainers.image.source="https://huggingface.co/arnavjain321/aasist-v3-codecaugment" \
      org.opencontainers.image.licenses="MIT AND BSD-2-Clause AND GPL-2.0-or-later AND Apache-2.0" \
      io.capstone.model.revision="${MODEL_REVISION}" \
      io.capstone.aasist.revision="${AASIST_REVISION}"

COPY --from=dependency-builder /opt/python /opt/python
COPY --from=dependency-builder /opt/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=dependency-builder /opt/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY --from=dependency-builder /opt/ffmpeg-package/imageio_ffmpeg-0.6.0.dist-info/LICENSE /var/task/licenses/IMAGEIO-FFMPEG-LICENSE

ENV PYTHONPATH=/opt/python:/var/task \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FFMPEG_BINARY=/usr/local/bin/ffmpeg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /var/task
COPY --from=artifact-verifier /source/artifacts/aasist_v3_best.pt ./artifacts/aasist_v3_best.pt
COPY --from=artifact-verifier /source/vendor/aasist/models/AASIST.py ./vendor/aasist/models/AASIST.py
COPY --from=artifact-verifier /source/vendor/aasist/LICENSE ./licenses/AASIST-LICENSE
COPY --from=artifact-verifier /source/vendor/aasist/NOTICE ./licenses/AASIST-NOTICE
COPY THIRD_PARTY_NOTICES.md ./licenses/THIRD_PARTY_NOTICES.md
COPY aasist_inference/ ./aasist_inference/
COPY inference_contract.py model_loader.py lambda_handler.py ./
COPY container-entrypoint.sh /usr/local/bin/container-entrypoint

RUN chmod 0555 /usr/local/bin/container-entrypoint \
    && python -c "from model_loader import load_model; model, metadata = load_model(); assert metadata['epoch'] == 19"

ENTRYPOINT ["/usr/local/bin/container-entrypoint"]
CMD ["lambda_handler.handler"]

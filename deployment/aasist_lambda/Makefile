IMAGE_NAME ?= aasist-lambda:phase5.5
PLATFORM ?= linux/amd64
CONTAINER_NAME ?= aasist-lambda-phase55
DEMO_ALLOWED_ORIGIN ?= http://127.0.0.1:8765
DEMO_PASSPHRASE ?=

.PHONY: container-build container-run container-stop container-test container-benchmark demo-local

container-build:
	docker buildx build --platform $(PLATFORM) --provenance=false --load --tag $(IMAGE_NAME) .

container-run:
	docker run --rm --detach --name $(CONTAINER_NAME) --platform $(PLATFORM) \
		--memory 2048m --cpus 1 \
		--read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
		--env DEMO_ALLOWED_ORIGIN=$(DEMO_ALLOWED_ORIGIN) \
		--env DEMO_PASSPHRASE=$(DEMO_PASSPHRASE) \
		--publish 9000:8080 $(IMAGE_NAME)

container-stop:
	-docker stop $(CONTAINER_NAME)

container-test:
	.venv/bin/python scripts/test_container_endpoint.py

container-benchmark:
	.venv/bin/python scripts/benchmark_container.py

demo-local:
	.venv/bin/python scripts/serve_demo_client.py

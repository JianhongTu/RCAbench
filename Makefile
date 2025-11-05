SHELL := /bin/bash
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

DOCKERHUB_USER := $(shell sed -n 's/^DOCKERHUB_USER=//p' src/rcabench/.env)
IMAGE := $(DOCKERHUB_USER)/rcabench:latest

.PHONY: deploy teardown secret logs port-forward build push run-local

deploy: secret
	@echo "Applying k8s manifests..."
	kubectl apply -f $(ROOT_DIR)/k8s/rcabench-deployment.yaml
	kubectl apply -f $(ROOT_DIR)/k8s/rcabench-service.yaml
	kubectl apply -f $(ROOT_DIR)/k8s/rcabench-ingress.yaml
	@echo "Deployed. Use 'make logs' or 'kubectl get pods -l app=rcabench'"

secret:
	@set -a; source src/rcabench/.env; set +a; \
	if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_API_KEY not set in .env"; exit 1; \
	fi; \
	$(ROOT_DIR)/scripts/create-secret.sh

teardown:
	@$(ROOT_DIR)/scripts/teardown.sh

logs:
	@kubectl logs -l app=rcabench -c gateway --tail=200 -f

port-forward:
	@kubectl port-forward svc/rcabench 8080:80

build:
	docker build -t $(IMAGE) -f docker/Dockerfile .

push:
	docker push $(IMAGE)

run-local: build
	docker run -p 8080:8080 --env-file src/rcabench/.env $(IMAGE)

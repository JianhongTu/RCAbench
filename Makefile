SHELL := /bin/bash
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

IMAGE := tovitu/rcabench:latest

.PHONY: deploy teardown secret logs port-forward build push

deploy: secret
	@echo "Applying k8s manifests..."
	kubectl apply -f $(ROOT_DIR)/k8s/gateway-deployment.yaml
	kubectl apply -f $(ROOT_DIR)/k8s/gateway-service.yaml
	kubectl apply -f $(ROOT_DIR)/k8s/gateway-ingress.yaml
	@echo "Deployed. Use 'make logs' or 'kubectl get pods -l app=rcabench'"

secret:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_API_KEY not set. Export it before running 'make deploy' or run scripts/create-secret.sh"; exit 1; \
	fi
	@$(ROOT_DIR)/scripts/create-secret.sh

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

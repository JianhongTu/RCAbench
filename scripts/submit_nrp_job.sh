#!/bin/bash
# Submit RCAbench pipeline job to NRP Kubernetes cluster (build from source)

TAG=${1:-$(date +%Y%m%d-%H%M%S)}
REPO_URL=${2:-"https://github.com/JianhongTu/RCAbench.git"}  # Default to JianhongTu/RCAbench
NAMESPACE=${3:-""}  # Optional namespace override
PVC_NAME=${4:-""}  # Optional PVC name override

echo "Submitting RCAbench pipeline job with tag: $TAG"
echo "Repository: $REPO_URL"
echo ""

# Detect namespace from current context if not provided
if [ -z "$NAMESPACE" ]; then
    NAMESPACE=$(kubectl config view --minify -o jsonpath='{.contexts[0].context.namespace}' 2>/dev/null)
    if [ -z "$NAMESPACE" ]; then
        # Try to get from current context name
        CURRENT_CTX=$(kubectl config current-context 2>/dev/null)
        NAMESPACE=$(kubectl config view -o jsonpath="{.contexts[?(@.name==\"$CURRENT_CTX\")].context.namespace}" 2>/dev/null)
    fi
    if [ -z "$NAMESPACE" ]; then
        # Try to get from context output (5th column)
        NAMESPACE=$(kubectl config get-contexts --no-headers | grep '^\*' | awk '{print $5}' 2>/dev/null)
    fi
fi

if [ -z "$NAMESPACE" ]; then
    NAMESPACE=$(kubectl config view --minify -o jsonpath='{.contexts[0].context.namespace}' 2>/dev/null)
    if [ -z "$NAMESPACE" ]; then
        CURRENT_CTX=$(kubectl config current-context 2>/dev/null)
        NAMESPACE=$(kubectl config view -o jsonpath="{.contexts[?(@.name==\"$CURRENT_CTX\")].context.namespace}" 2>/dev/null)
    fi
    if [ -z "$NAMESPACE" ]; then
        NAMESPACE=$(kubectl config get-contexts --no-headers | grep '^\*' | awk '{print $5}' 2>/dev/null)
    fi
fi

if [ -n "$NAMESPACE" ]; then
    echo "Using namespace: $NAMESPACE"
    NAMESPACE_FLAG="-n $NAMESPACE"
else
    echo "⚠️  No namespace detected in kubeconfig"
    echo "   Attempting to use default namespace (this may fail if you don't have permissions)"
    echo "   To specify a namespace, run: $0 $TAG $REPO_URL <namespace> [pvc-name]"
    echo "   Example: $0 $TAG $REPO_URL wang-research-lab gaurs-storage"
    NAMESPACE_FLAG=""
    echo ""
fi

# Detect PVC name if not provided (default to gaurs-storage)
if [ -z "$PVC_NAME" ]; then
    PVC_NAME="gaurs-storage"
    # Verify it exists
    if [ -n "$NAMESPACE" ] && ! kubectl get pvc "$PVC_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
        # Try to find a PVC that matches common patterns
        PVC_NAME=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null | head -1)
        if [ -z "$PVC_NAME" ]; then
            # Try to find one with "storage" in the name
            PVC_NAME=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -i storage | head -1)
        fi
    fi
fi

PVC_TO_USE=${PVC_NAME:-gaurs-storage}

if [ -z "$PVC_TO_USE" ]; then
    echo "⚠️  No PVC detected. Results will not be persisted."
    echo "   To specify a PVC, run: $0 $TAG $REPO_URL <namespace> <pvc-name>"
    echo "   Example: $0 $TAG $REPO_URL wang-research-lab gaurs-storage"
    echo "   Available PVCs:"
    if [ -n "$NAMESPACE" ]; then
        kubectl get pvc -n "$NAMESPACE" 2>/dev/null | head -5 || echo "   (cannot list PVCs)"
    fi
    USE_PVC=0
else
    echo "Using PVC: $PVC_TO_USE"
    USE_PVC=1
fi
echo ""

# Function to retry kubectl commands with exponential backoff
# Note: Does not retry on permission errors (Forbidden)
kubectl_retry() {
    local max_attempts=3
    local attempt=1
    local delay=2
    
    while [ $attempt -le $max_attempts ]; do
        local output
        local exit_code
        
        # Capture both output and exit code
        output=$(kubectl "$@" --request-timeout=60s 2>&1)
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            # Return the command output to stdout so callers can capture it
            if [ -n "$output" ]; then
                printf '%s\n' "$output"
            fi
            return 0
        fi
        
        # Don't retry on permission errors - they won't be fixed by retrying
        if echo "$output" | grep -q "Forbidden\|forbidden"; then
            echo "$output" >&2
            echo "❌ Permission denied - this error won't be fixed by retrying" >&2
            return $exit_code
        fi
        
        # Don't retry on validation errors
        if echo "$output" | grep -q "error validating\|Invalid\|invalid"; then
            echo "$output" >&2
            return $exit_code
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            echo "⚠️ kubectl command failed (attempt $attempt/$max_attempts). Retrying in ${delay}s..." >&2
            echo "$output" >&2
            sleep $delay
            delay=$((delay * 2))
        else
            echo "$output" >&2
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "❌ kubectl command failed after $max_attempts attempts" >&2
    return 1
}

# Verify cluster connectivity before proceeding
echo "Verifying cluster connectivity..."
# Use a simple API call that doesn't require special permissions
if ! kubectl_retry get --raw /api/v1 >/dev/null 2>&1; then
    echo "❌ Cannot connect to Kubernetes cluster. Please check:"
    echo "   1. kubectl is configured correctly (kubectl config get-contexts)"
    echo "   2. Network connectivity to the cluster"
    echo "   3. Cluster API server is reachable"
    echo ""
    echo "Run diagnostics: ./scripts/check_cluster_connectivity.sh"
    exit 1
fi
echo "✅ Cluster connectivity verified"
echo ""

# Replace ${TAG} variable and apply the job
export TAG
TEMP_MANIFEST=$(mktemp)
envsubst < k8s/rcabench-pipeline-job-build.yml > "${TEMP_MANIFEST}"

# Add namespace to manifest if we have one
if [ -n "$NAMESPACE" ]; then
    # Use yq if available, otherwise use awk (more reliable than sed on macOS)
    if command -v yq >/dev/null 2>&1; then
        yq eval ".metadata.namespace = \"$NAMESPACE\"" -i "${TEMP_MANIFEST}"
    else
        # Use awk to insert namespace after the name line (works reliably on macOS and Linux)
        awk -v ns="$NAMESPACE" '/^  name: / {print; print "  namespace: " ns; next} 1' "${TEMP_MANIFEST}" > "${TEMP_MANIFEST}.tmp" && mv "${TEMP_MANIFEST}.tmp" "${TEMP_MANIFEST}"
    fi
fi

# Update PVC name in manifest (defaults to gaurs-storage, can be overridden)
PVC_TO_USE=${PVC_NAME:-gaurs-storage}
if command -v yq >/dev/null 2>&1; then
    yq eval ".spec.template.spec.volumes[] |= (select(.name == \"results-pvc\") | .persistentVolumeClaim.claimName = \"$PVC_TO_USE\")" -i "${TEMP_MANIFEST}" 2>/dev/null || true
else
    # Use sed to replace PVC name (handles both existing names)
    sed -i.bak "s/claimName:.*/claimName: $PVC_TO_USE/" "${TEMP_MANIFEST}" 2>/dev/null || true
    rm -f "${TEMP_MANIFEST}.bak" 2>/dev/null || true
fi

# Check if job already exists - if so, delete it first (job spec.template is immutable)
JOB_NAME="rcabench-pipeline-${TAG}"
if kubectl get job ${JOB_NAME} $NAMESPACE_FLAG >/dev/null 2>&1; then
    echo "Job ${JOB_NAME} already exists. Deleting it first..."
    kubectl delete job ${JOB_NAME} $NAMESPACE_FLAG --wait=false
    echo "Waiting for job to be deleted..."
    for i in {1..30}; do
        if ! kubectl get job ${JOB_NAME} $NAMESPACE_FLAG >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi

if ! kubectl_retry apply -f "${TEMP_MANIFEST}" $NAMESPACE_FLAG; then
    rm -f "${TEMP_MANIFEST}"
    echo "❌ Failed to apply job manifest"
    if [ -z "$NAMESPACE" ]; then
        echo "💡 Tip: You may need to specify a namespace. Check your context:"
        echo "   kubectl config get-contexts"
        echo "   Then run: $0 $TAG $REPO_URL <namespace>"
    fi
    exit 1
fi
rm -f "${TEMP_MANIFEST}"

# JOB_NAME already set above

echo ""
echo "✅ Job submitted: ${JOB_NAME}"
echo ""

# Wait for pod creation
echo "Waiting for pod to be scheduled..."
for i in {1..60}; do
    POD_NAME=$(kubectl_retry get pods -l job-name=${JOB_NAME} $NAMESPACE_FLAG -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$POD_NAME" ]; then
        break
    fi
    sleep 5
done

if [ -z "$POD_NAME" ]; then
    echo "❌ Could not find pod for job ${JOB_NAME}. Check with: kubectl get pods $NAMESPACE_FLAG"
    exit 1
fi

echo "Pod: ${POD_NAME}"
echo ""
echo "Streaming logs (Ctrl+C to stop). They continue in background..."
kubectl logs -f job/${JOB_NAME} $NAMESPACE_FLAG --request-timeout=60s &
LOGS_PID=$!

echo ""
echo "Waiting for results to be generated and job to complete..."

RESULTS_DIR="./results/${TAG}"
mkdir -p "${RESULTS_DIR}"

# Function to copy results - only works on Running pods
copy_results() {
    local pod_name=$1
    if [ -z "$pod_name" ]; then
        return 1
    fi
    
    # Use kubectl cp (only works on Running pods)
    if [ -n "$NAMESPACE" ]; then
        CP_SOURCE="${NAMESPACE}/${pod_name}:/workspace/rcabench/tmp/validation"
    else
        CP_SOURCE="${pod_name}:/workspace/rcabench/tmp/validation"
    fi
    
    if kubectl cp "$CP_SOURCE" "${RESULTS_DIR}/validation" 2>/dev/null; then
        return 0
    fi
    
    return 1
}

# Wait for job to complete, then copy from PVC
# Results are stored in PVC, so we can copy even after pod completes
echo ""
echo "Waiting for job to complete..."
# Use kubectl wait directly - suppress stderr as it may show warnings during pod initialization
kubectl wait --for=condition=complete job/${JOB_NAME} $NAMESPACE_FLAG --timeout=3600s 2>/dev/null || {
    echo "⚠️ Job did not complete within timeout. Checking status..."
    kubectl get job ${JOB_NAME} $NAMESPACE_FLAG
}

echo ""
echo "Job completed. Copying results from PVC..."

# Copy from PVC using a temporary pod (only if PVC is configured)
if [ "$USE_PVC" -eq 1 ] && [ -n "$PVC_TO_USE" ]; then
    PVC_POD_NAME="pvc-copy-${TAG}-$(date +%s)"
    RESULTS_COPIED=0

    # Create a temporary pod to access the PVC
    cat <<EOF | kubectl apply $NAMESPACE_FLAG -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${PVC_POD_NAME}
spec:
  containers:
  - name: copy
    image: busybox
    command: ["sleep", "300"]
    volumeMounts:
    - name: results-pvc
      mountPath: /results
  volumes:
  - name: results-pvc
    persistentVolumeClaim:
      claimName: ${PVC_TO_USE}
  tolerations:
  - key: "nautilus.io/chase-ci"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  restartPolicy: Never
EOF

    # Wait for pod to be ready
    echo "Waiting for PVC access pod to be ready..."
    POD_READY=0
    for i in {1..60}; do
        POD_PHASE=$(kubectl get pod ${PVC_POD_NAME} $NAMESPACE_FLAG -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$POD_PHASE" = "Running" ]; then
            POD_READY=1
            break
        elif [ "$POD_PHASE" = "Failed" ] || [ "$POD_PHASE" = "Error" ]; then
            echo "⚠️ PVC access pod failed to start. Status: ${POD_PHASE}"
            kubectl describe pod ${PVC_POD_NAME} $NAMESPACE_FLAG | tail -20
            break
        fi
        sleep 2
    done

    if [ "$POD_READY" -eq 0 ]; then
        echo "⚠️ PVC access pod did not become ready. Cannot copy results."
        kubectl get pod ${PVC_POD_NAME} $NAMESPACE_FLAG
    else
        # Copy results from PVC
        echo "PVC pod is ready. Checking for results..."
        if kubectl exec ${PVC_POD_NAME} $NAMESPACE_FLAG -- test -d /results/${TAG}/validation 2>/dev/null; then
        echo "Copying results from PVC..."
        if kubectl cp ${NAMESPACE:+$NAMESPACE/}${PVC_POD_NAME}:/results/${TAG}/validation "${RESULTS_DIR}/validation" $NAMESPACE_FLAG 2>/dev/null; then
            echo "✅ Results copied to ${RESULTS_DIR}/validation"
            RESULTS_COPIED=1
        else
            echo "⚠️ Failed to copy from PVC pod"
        fi
        else
            echo "⚠️ Results directory not found in PVC at /results/${TAG}/validation"
            echo "   Checking what's in the PVC..."
            kubectl exec ${PVC_POD_NAME} $NAMESPACE_FLAG -- ls -la /results/ 2>/dev/null || echo "   (cannot list PVC contents)"
            kubectl exec ${PVC_POD_NAME} $NAMESPACE_FLAG -- find /results -name "*.json" -type f 2>/dev/null | head -5 || echo "   (no JSON files found)"
        fi
    fi

    # Clean up temporary pod
    kubectl delete pod ${PVC_POD_NAME} $NAMESPACE_FLAG --ignore-not-found=true >/dev/null 2>&1
else
    echo "⚠️ No PVC configured. Cannot copy results from completed pod."
    echo "   Results may be lost if pod has already completed."
    RESULTS_COPIED=0
fi

# Stop log streaming
kill ${LOGS_PID} >/dev/null 2>&1 || true

if [ "${RESULTS_COPIED}" -eq 1 ]; then
    echo ""
    echo "Results location: ${RESULTS_DIR}/validation/"
    ls -lh "${RESULTS_DIR}/validation/" 2>/dev/null | head -10 || echo "  listing contents..."
else
    echo ""
    echo "⚠️ Failed to copy results automatically."
    if [ -n "$POD_NAME" ]; then
        if [ -n "$NAMESPACE" ]; then
            CP_SOURCE="${NAMESPACE}/${POD_NAME}:/workspace/rcabench/tmp/validation"
        else
            CP_SOURCE="${POD_NAME}:/workspace/rcabench/tmp/validation"
        fi
        echo "   Try manually with:"
        echo "   kubectl cp $CP_SOURCE ./results/${TAG}/validation"
    else
        echo "   Pod may have been cleaned up. Check job status:"
        echo "   kubectl get job ${JOB_NAME} $NAMESPACE_FLAG"
    fi
fi

echo ""
echo "When done, you can delete the job with:"
echo "  kubectl delete job ${JOB_NAME} $NAMESPACE_FLAG"
echo ""


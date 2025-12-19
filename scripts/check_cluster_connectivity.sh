#!/bin/bash
# Diagnostic script to check Kubernetes cluster connectivity

echo "🔍 Kubernetes Cluster Connectivity Diagnostics"
echo "=============================================="
echo ""

# 1. Check kubectl configuration and contexts
echo "=== 1. Checking kubectl configuration ==="
if ! command -v kubectl >/dev/null 2>&1; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi
echo "✅ kubectl is installed: $(kubectl version --client --short 2>/dev/null || echo 'version unknown')"
echo ""
echo "Available contexts:"
kubectl config get-contexts
echo ""
CURRENT_CTX=$(kubectl config current-context 2>/dev/null)
if [ -n "$CURRENT_CTX" ]; then
    echo "Current context: $CURRENT_CTX"
else
    echo "❌ No current context set"
fi
echo ""

# 2. Check if kubectl can reach the API server
echo "=== 2. Testing API server connectivity ==="
CLUSTER_INFO_OUTPUT=$(kubectl cluster-info --request-timeout=10s 2>&1)
CLUSTER_INFO_EXIT=$?
if [ $CLUSTER_INFO_EXIT -eq 0 ]; then
    echo "✅ Successfully connected to cluster"
    echo "$CLUSTER_INFO_OUTPUT"
elif echo "$CLUSTER_INFO_OUTPUT" | grep -q "Forbidden"; then
    echo "⚠️  Connected but permission denied (this may be normal for limited users)"
    echo "$CLUSTER_INFO_OUTPUT"
elif echo "$CLUSTER_INFO_OUTPUT" | grep -q "timeout\|deadline exceeded"; then
    echo "❌ Connection timeout - API server is not reachable"
    echo "$CLUSTER_INFO_OUTPUT"
else
    echo "❌ Failed to connect to cluster"
    echo "$CLUSTER_INFO_OUTPUT"
fi
echo ""

# 3. Check network connectivity to the API server
echo "=== 3. Checking network connectivity ==="
API_SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null)
if [ -n "$API_SERVER" ]; then
    echo "API Server URL: $API_SERVER"
    # Extract hostname and port (handle https://host:port format)
    HOST_PORT=$(echo "$API_SERVER" | sed -E 's|^https?://||' | cut -d'/' -f1)
    HOST=$(echo "$HOST_PORT" | cut -d':' -f1)
    PORT=$(echo "$HOST_PORT" | cut -d':' -f2)
    PORT=${PORT:-443}
    echo "Testing connection to $HOST:$PORT..."
    
    # Use bash built-in for port testing (works on macOS)
    if (bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null) && exec 3<&- && exec 3>&-; then
        echo "✅ Port $PORT is reachable on $HOST"
    elif command -v nc >/dev/null 2>&1; then
        # Try with nc if available (macOS has nc but may need -G flag for timeout)
        if nc -z -G 5 "$HOST" "$PORT" 2>/dev/null; then
            echo "✅ Port $PORT is reachable on $HOST"
        else
            echo "❌ Cannot connect to $HOST:$PORT"
            echo "   This could indicate:"
            echo "   - Firewall blocking the connection"
            echo "   - VPN not connected"
            echo "   - Network routing issues"
            echo "   - API server is down"
        fi
    else
        echo "⚠️  Cannot test port connectivity (nc not available)"
        echo "   Try manually: curl -k -v $API_SERVER"
    fi
else
    echo "❌ Could not determine API server URL from kubeconfig"
fi
echo ""

# 4. Check authentication
echo "=== 4. Testing authentication ==="
if kubectl auth can-i get pods --all-namespaces --request-timeout=10s 2>&1 | grep -q "yes"; then
    echo "✅ Authentication successful"
else
    echo "⚠️  Authentication check failed or timed out"
    echo "   This could indicate:"
    echo "   - Expired credentials"
    echo "   - Invalid kubeconfig"
    echo "   - Network timeout preventing auth"
fi
echo ""

# 5. Check API server response time
echo "=== 5. Testing API server response time ==="
if kubectl get --raw /api/v1 --request-timeout=10s >/dev/null 2>&1; then
    echo "✅ API server is responding"
    echo "Response time test:"
    time kubectl get --raw /api/v1 --request-timeout=10s >/dev/null 2>&1
else
    echo "❌ API server is not responding or request timed out"
    echo "   Trying with verbose output:"
    kubectl get --raw /api/v1 --request-timeout=10s 2>&1 | head -5
fi
echo ""

# 6. Additional diagnostics
echo "=== 6. Additional diagnostics ==="
echo "Kubeconfig file location:"
echo "  KUBECONFIG env: ${KUBECONFIG:-<not set, using default ~/.kube/config>}"
echo ""
echo "To manually test connection:"
echo "  kubectl get nodes --request-timeout=10s"
echo "  kubectl get namespaces --request-timeout=10s"


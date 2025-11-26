#!/bin/bash
# Cleanup script for patch-verify Kubernetes jobs
# Usage: ./cleanup_patch_jobs.sh [namespace] [options]

set -e

# Default namespace
NAMESPACE="${1:-wang-research-lab}"

echo "=== CLEANUP PATCH-VERIFY JOBS ==="
echo "Namespace: $NAMESPACE"
echo ""

# Function to cleanup jobs
cleanup_jobs() {
    local filter="$1"
    local description="$2"

    echo "Finding $description jobs..."
    local jobs=$(kubectl get jobs -n "$NAMESPACE" --no-headers | grep "^patch-verify" | awk '{print $1}' | tr '\n' ' ')

    if [ -z "$jobs" ]; then
        echo "No $description jobs found."
        return
    fi

    echo "Found jobs: $jobs"
    echo ""

    # Count jobs
    local job_count=$(echo "$jobs" | wc -w)
    echo "Found $job_count $description job(s)"
    echo ""

    # Ask for confirmation
    read -p "Delete these $job_count $description job(s)? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping deletion of $description jobs."
        return
    fi

    # Delete jobs
    echo "Deleting $description jobs..."
    for job in $jobs; do
        echo "Deleting job: $job"
        kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found=true
    done
    echo "✅ Deleted $job_count $description job(s)"
    echo ""
}

# Show current jobs status
echo "Current patch-verify jobs status:"
kubectl get jobs -n "$NAMESPACE" | grep patch-verify || echo "No patch-verify jobs found."
echo ""

# Option to cleanup all jobs
echo "Choose cleanup option:"
echo "1) Delete ALL patch-verify jobs"
echo "2) Delete only COMPLETED jobs"
echo "3) Delete only FAILED jobs"
echo "4) Show detailed job status"
echo "5) Exit"
echo ""

read -p "Enter choice (1-5): " choice
echo ""

case $choice in
    1)
        echo "🧹 Deleting ALL patch-verify jobs..."
        jobs_to_delete=$(kubectl get jobs -n "$NAMESPACE" --no-headers | grep "^patch-verify" | awk '{print $1}' | tr '\n' ' ')
        if [ -n "$jobs_to_delete" ]; then
            kubectl delete jobs $jobs_to_delete -n "$NAMESPACE" --ignore-not-found=true
            echo "✅ All patch-verify jobs deleted."
        else
            echo "No patch-verify jobs found."
        fi
        ;;
    2)
        echo "🧹 Deleting COMPLETED patch-verify jobs..."
        # Get completed jobs and delete them
        completed_jobs=$(kubectl get jobs -n "$NAMESPACE" --no-headers | grep "^patch-verify.*1/1" | awk '{print $1}' | tr '\n' ' ')
        if [ -n "$completed_jobs" ]; then
            echo "Found completed jobs: $completed_jobs"
            kubectl delete jobs $completed_jobs -n "$NAMESPACE" --ignore-not-found=true
            echo "✅ Completed jobs deleted."
        else
            echo "No completed patch-verify jobs found."
        fi
        ;;
    3)
        echo "🧹 Deleting FAILED patch-verify jobs..."
        # Get failed jobs (jobs that have failed, not just running)
        failed_jobs=$(kubectl get jobs -n "$NAMESPACE" --no-headers | grep "^patch-verify.*Failed" | awk '{print $1}' | tr '\n' ' ')
        if [ -n "$failed_jobs" ]; then
            echo "Found failed jobs: $failed_jobs"
            kubectl delete jobs $failed_jobs -n "$NAMESPACE" --ignore-not-found=true
            echo "✅ Failed jobs deleted."
        else
            echo "No failed patch-verify jobs found."
        fi
        ;;
    4)
        echo "📊 Detailed job status:"
        kubectl get jobs -n "$NAMESPACE" | grep patch-verify
        echo ""
        echo "📋 Pod status for patch-verify jobs:"
        kubectl get pods -n "$NAMESPACE" | grep patch-verify
        ;;
    5)
        echo "Exiting without cleanup."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== CLEANUP COMPLETE ==="
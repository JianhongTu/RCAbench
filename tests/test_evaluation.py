#!/usr/bin/env python3

import sys
sys.path.append('/home/tovitu/codes/RCAbench/src')

from rcabench.server.eval_utils import get_ground_truth, evaluate_localization

if __name__ == "__main__":
    arvo_id = "10013"
    try:
        gt = get_ground_truth(arvo_id)
        print("Ground truth:")
        for loc in gt:
            print(f"  {loc.file}: old {loc.old_span.start}-{loc.old_span.end}, new {loc.new_span.start}-{loc.new_span.end}")

        # 2. Create test list identical to gt
        preds = gt.copy()
        report = evaluate_localization(preds, gt)
        print("\nEvaluation with identical preds:")
        print(f"Task ID: {report.task_id}")
        print(f"N GT: {report.n_gt}, N Pred: {report.n_pred}")
        print(f"File Acc: {report.file_acc}")
        print(f"Func TopK Recall: {report.func_topk_recall}")
        print(f"Line TopK Recall: {report.line_topk_recall}")
        print(f"Line IoU Mean: {report.line_iou_mean}")

        # 3. Modify to introduce errors
        if preds:
            # Change file to wrong one
            preds[0] = preds[0].model_copy(update={"file": "wrong.c"})
        report2 = evaluate_localization(preds, gt)
        print("\nEvaluation with modified preds (wrong file):")
        print(f"Task ID: {report2.task_id}")
        print(f"N GT: {report2.n_gt}, N Pred: {report2.n_pred}")
        print(f"File Acc: {report2.file_acc}")
        print(f"Func TopK Recall: {report2.func_topk_recall}")
        print(f"Line TopK Recall: {report2.line_topk_recall}")
        print(f"Line IoU Mean: {report2.line_iou_mean}")

    except Exception as e:
        print(f"Error: {e}")

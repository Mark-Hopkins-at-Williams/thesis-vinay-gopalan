from transformers.data.metrics import simple_accuracy

def sst3_compute_metrics(task_name, preds, labels):
    """Metrics for computing SST-3 task."""
    assert len(preds) == len(labels)
    if task_name == "sst-3":
        return {
                "pred": preds,
                "actual": labels
                }
    else:
        raise KeyError(task_name)
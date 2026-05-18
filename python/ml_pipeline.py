"""Legacy compatibility helpers kept for existing tests and scripts."""

from python.pcxp_mlops.config import get_paths
from python.pcxp_mlops.data_loader import count_images_by_label
from python.pcxp_mlops.metrics import apply_threshold, recall_from_confusion_matrix


def build_project_paths(current_dir):
    return get_paths(current_dir).as_dict()

from pathlib import Path
from typing import Dict

from .base import MetricBase
from .depth import DepthMetric
from .optical_flow import OpticFlowMetric
from .semantic import SegmentationMetric
from .panoptic import PanopticMetric
from .object_detection import BoundaryBoxMetric
from .object_detection import ClassificationMetric

def get_loggers(logger_cfg: Dict[str, str], basepath: Path) -> Dict[str, MetricBase]:
    """
    Given a dictionary of [key, value] = [objective type, main metric] and
    basepath to save the file returns a dictionary that consists of performance
    metric trackers.
    """
    loggers = {}

    for logger_type, main_metric in logger_cfg.items():
        logger_args = dict(main_metric=main_metric, base_dir=basepath)
        if logger_type == 'flow':
            loggers['flow'] = OpticFlowMetric()
        elif logger_type == 'seg':
            loggers['seg'] = SegmentationMetric(19, **logger_args)
        elif logger_type == 'depth':
            loggers['depth'] = DepthMetric(**logger_args)
        elif logger_type == 'bbox':
            loggers['bbox'] = BoundaryBoxMetric(19, **logger_args)
        elif logger_type == 'panoptic':
            loggers['panoptic'] = PanopticMetric(19, **logger_args)
        else:
            raise NotImplementedError(logger_type)

    return loggers

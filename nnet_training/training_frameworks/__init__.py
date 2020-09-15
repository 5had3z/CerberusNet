from .trainer_base_class import ModelTrainer
from .mono_flow_trainer import MonoFlowTrainer
from .mono_seg_flow_trainer import MonoSegFlowTrainer
from .mono_seg_trainer import MonoSegmentationTrainer
from .stereo_depth_trainer import StereoDisparityTrainer
from .stereo_seg_depth_trainer import StereoSegDepthTrainer
from .stereo_seg_trainer import StereoSegTrainer
from .stereo_flow_trainer import StereoFlowTrainer
from .mono_seg_flow_depth_trainer import MonoSegFlowDepthTrainer

def get_trainer(trainer_name: str) -> ModelTrainer:
    """
    Returns the corresponding network trainer given a string
    """

    if trainer_name == "MonoFlowTrainer":
        trainer = MonoFlowTrainer
    elif trainer_name == "MonoSegFlowTrainer":
        trainer = MonoSegFlowTrainer
    elif trainer_name == "MonoSegmentationTrainer":
        trainer = MonoSegmentationTrainer
    elif trainer_name == "StereoDisparityTrainer":
        trainer = StereoDisparityTrainer
    elif trainer_name == "StereoFlowTrainer":
        trainer = StereoFlowTrainer
    elif trainer_name == "StereoSegDepthTrainer":
        trainer = StereoSegDepthTrainer
    elif trainer_name == "StereoSegTrainer":
        trainer = StereoSegTrainer
    elif trainer_name == "MonoSegFlowDepthTrainer":
        trainer = MonoSegFlowDepthTrainer
    else:
        raise NotImplementedError(trainer_name)

    return trainer

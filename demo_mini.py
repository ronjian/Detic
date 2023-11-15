import sys
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer

# constants
WINDOW_NAME = "Detic"

def setup_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file('configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

class Args():
    vocabulary = 'lvis'

if __name__ == "__main__":
    cfg = setup_cfg()
    detic_demo = VisualizationDemo(cfg, Args())
    img = read_image('./desk.jpg', format="BGR")
    predictions = detic_demo.predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], detic_demo.metadata, instance_mode=detic_demo.instance_mode)
    predictions_instances = predictions["instances"].to(detic_demo.cpu_device)
    vis_output = visualizer.draw_instance_predictions(predictions=predictions_instances)
    vis_output.save('./out.png')

    # 找出目标类别的MASK
    classes_name = detic_demo.metadata.get("thing_classes", None)
    scores = predictions_instances.scores
    classes = predictions_instances.pred_classes.tolist()
    masks = np.asarray(predictions_instances.pred_masks)
    target_index = classes_name.index('chair')
    for class_index, score, mask in zip(classes, scores, masks):
        if class_index == target_index:
            print(class_index)
            print(score)
            print(mask.shape)

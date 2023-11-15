import sys
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer

import rospy
import sensor_msgs
from cv_bridge import CvBridge

COLOR_IMAGE=None

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

def color_image_callback(msg):
    # 创建CvBridge对象
    bridge = CvBridge()
    # 将ROS图像消息转换为OpenCV图像格式
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    # 将OpenCV图像转换为NumPy数组
    global COLOR_IMAGE
    COLOR_IMAGE = np.array(cv_image)[:, :, ::-1].copy() # bgr的通道顺序, 0-255
    return

def demo(event):
    if COLOR_IMAGE is None:
        return
    cfg = setup_cfg()
    detic_demo = VisualizationDemo(cfg, Args())
    # img = read_image('./desk.jpg', format="BGR")
    img = COLOR_IMAGE
    predictions = detic_demo.predictor(img)
    predictions_instances = predictions["instances"].to(detic_demo.cpu_device)
    visualizer = Visualizer(img[:, :, ::-1], detic_demo.metadata, instance_mode=detic_demo.instance_mode)
    vis_output = visualizer.draw_instance_predictions(predictions=predictions_instances)
    vis_output.save('./out.png')

    # 找出目标类别的MASK
    classes_name = detic_demo.metadata.get("thing_classes", None)
    scores = predictions_instances.scores
    classes = predictions_instances.pred_classes.tolist()
    masks = np.asarray(predictions_instances.pred_masks)
    target_index = classes_name.index('shoe')
    target_mask = np.zeros(masks[0].shape, dtype=bool)
    for class_index, score, mask in zip(classes, scores, masks):
        if class_index == target_index:
            print(class_index)
            print(score)
            print(mask.shape)
            target_mask = target_mask | mask
    publish_image(target_mask)
    return

def publish_image(bool_array):
    # 创建CvBridge对象
    bridge = CvBridge()
    
    # 创建sensor_msgs/Image消息
    image_msg = sensor_msgs.msg.Image()
    image_msg.header.stamp = rospy.Time.now()
    image_msg.height = bool_array.shape[0]
    image_msg.width = bool_array.shape[1]
    image_msg.encoding = "mono8"
    image_msg.step = bool_array.shape[1]
    
    # 将布尔数组转换为uint8类型的NumPy数组
    uint8_array = np.asarray(bool_array, dtype=np.uint8)
    
    # 将NumPy数组转换为ROS图像消息
    image_msg.data = uint8_array.tostring()
    
    # 发布图像消息
    target_mask_publisher.publish(image_msg) 

if __name__ == "__main__":
    rospy.init_node('detic_ros_demo')

    _ = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, color_image_callback)
    target_mask_publisher = rospy.Publisher('/detic/target_mask', sensor_msgs.msg.Image, queue_size=10)
    _ = rospy.Timer(rospy.Duration(1.0), demo)

    # 进入ROS循环
    rospy.spin()
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the model
MODEL_PATH = './ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model' 
detect_fn = tf.saved_model.load(MODEL_PATH)

# Load the label map
PATH_TO_LABELS = './research/object_detection/data/mscoco_label_map.pbtxt' 
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load an image
IMAGE_PATH = './object.jpg'  
image_np = cv2.imread(IMAGE_PATH)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

# Perform object detection
detections = detect_fn(input_tensor)

# Extract detection data
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# Print detected labels and scores
print("\nDetected Objects:")
for i in range(num_detections):
    class_id = int(detections['detection_classes'][i])  
    score = detections['detection_scores'][i]          
    if score > 0.5: 
        label = category_index[class_id]['name']        
        print(f" - {label}: {score:.2%}")               

# Visualize detections
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'].astype(int),
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=0.5,
    agnostic_mode=False
)

# Display the image
cv2.imshow('Object Detection', cv2.resize(image_np_with_detections, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()

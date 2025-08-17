# utils/stateless_visualization.py - Product-aware visualization
"""
Product-aware visualization utilities for generating annotated images
Automatically detects product boundaries and applies appropriate borders
"""

import cv2
import numpy as np
import base64
import logging
from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES

logger = logging.getLogger(__name__)

def create_annotated_image_base64(image_path, result):
    """Create base64 encoded annotated image with product-aware borders"""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image for annotation: {image_path}")
            return None
        
        # Create annotated version with product-aware borders
        annotated_image = annotate_detection_result_product_aware(image, result)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return None

def annotate_detection_result_product_aware(image, result):
    """Add annotations to image with product-aware borders"""
    try:
        annotated_frame = image.copy()
        height, width = annotated_frame.shape[:2]
        
        if result:
            # Detect product boundaries
            product_bbox = detect_product_boundaries(image)
            
            if result['final_decision'] == 'GOOD':
                # Add green border around product area only
                if product_bbox:
                    draw_product_border(annotated_frame, product_bbox, (0, 255, 0), 5)
                else:
                    # Fallback to full image border if product detection fails
                    annotated_frame = cv2.copyMakeBorder(annotated_frame, 5, 5, 5, 5, 
                                                       cv2.BORDER_CONSTANT, value=(0, 255, 0))
                
                cv2.putText(annotated_frame, "GOOD", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                           
            elif result['final_decision'] == 'DEFECT':
                # Add red border around product area only
                if product_bbox:
                    draw_product_border(annotated_frame, product_bbox, (0, 0, 255), 8)
                else:
                    # Fallback to full image border if product detection fails
                    annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, 
                                                       cv2.BORDER_CONSTANT, value=(0, 0, 255))
                
                cv2.putText(annotated_frame, "DEFECT", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes
                if result.get('defect_classification') and 'bounding_boxes' in result['defect_classification']:
                    draw_defect_bounding_boxes(annotated_frame, result['defect_classification']['bounding_boxes'])
                elif (result.get('defect_classification') and 
                      result['defect_classification'].get('defect_analysis') and
                      'bounding_boxes' in result['defect_classification']['defect_analysis']):
                    draw_defect_bounding_boxes(
                        annotated_frame, 
                        result['defect_classification']['defect_analysis']['bounding_boxes']
                    )
            
            # Add processing info
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            cv2.putText(annotated_frame, f"Score: {anomaly_score:.3f}", 
                       (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add processing mode info
            processing_mode = result.get('processing_mode', 'standard')
            cv2.putText(annotated_frame, f"Mode: {processing_mode}", 
                       (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"Error annotating detection result: {e}")
        return image

def detect_product_boundaries(image):
    """Detect product boundaries in the image using edge detection and contour analysis"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with adaptive thresholds
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by size and aspect ratio
        valid_contours = []
        min_area = (width * height) * 0.05  # At least 5% of image
        max_area = (width * height) * 0.8   # At most 80% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check if contour is not too close to image edges
                x, y, w, h = cv2.boundingRect(contour)
                
                # Avoid contours that touch image borders
                margin = min(width, height) * 0.02  # 2% margin
                if (x > margin and y > margin and 
                    x + w < width - margin and y + h < height - margin):
                    
                    # Check aspect ratio is reasonable
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                        valid_contours.append((contour, area))
        
        if not valid_contours:
            # Fallback: use center region of image
            margin_x = int(width * 0.1)
            margin_y = int(height * 0.1)
            return {
                'x': margin_x,
                'y': margin_y,
                'width': width - 2 * margin_x,
                'height': height - 2 * margin_y
            }
        
        # Select the largest valid contour
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Add small padding
        padding = min(w, h) * 0.05
        x = max(0, int(x - padding))
        y = max(0, int(y - padding))
        w = min(width - x, int(w + 2 * padding))
        h = min(height - y, int(h + 2 * padding))
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }
        
    except Exception as e:
        logger.error(f"Error detecting product boundaries: {e}")
        return None

def draw_product_border(image, product_bbox, color, thickness):
    """Draw border around product area"""
    try:
        x = product_bbox['x']
        y = product_bbox['y']
        w = product_bbox['width']
        h = product_bbox['height']
        
        # Draw thick border around product
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Optional: Add corner markers for better visibility
        corner_size = min(w, h) // 20
        
        # Top-left corner
        cv2.line(image, (x, y), (x + corner_size, y), color, thickness + 2)
        cv2.line(image, (x, y), (x, y + corner_size), color, thickness + 2)
        
        # Top-right corner
        cv2.line(image, (x + w, y), (x + w - corner_size, y), color, thickness + 2)
        cv2.line(image, (x + w, y), (x + w, y + corner_size), color, thickness + 2)
        
        # Bottom-left corner
        cv2.line(image, (x, y + h), (x + corner_size, y + h), color, thickness + 2)
        cv2.line(image, (x, y + h), (x, y + h - corner_size), color, thickness + 2)
        
        # Bottom-right corner
        cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), color, thickness + 2)
        cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), color, thickness + 2)
        
    except Exception as e:
        logger.error(f"Error drawing product border: {e}")

def draw_defect_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes for defects with product-aware styling"""
    try:
        for defect_type, bboxes in bounding_boxes.items():
            # Skip background class
            if defect_type == 'background':
                continue
            
            # Get color from config based on defect type
            defect_class_id = None
            for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
                if class_name == defect_type:
                    defect_class_id = class_id
                    break
            
            # Use config color if available, otherwise use default
            if defect_class_id is not None and defect_class_id in DEFECT_COLORS:
                color = DEFECT_COLORS[defect_class_id]
            else:
                # Fallback colors
                default_colors = {
                    'scratch': (0, 255, 255),     # Cyan
                    'stained': (128, 0, 128),     # Purple  
                    'damaged': (255, 0, 0),       # Red
                    'missing_component': (255, 255, 0),  # Yellow
                    'open': (255, 0, 255),        # Magenta
                }
                color = default_colors.get(defect_type, (255, 255, 255))  # White default
            
            for bbox in bboxes:
                x, y = bbox['x'], bbox['y']
                w, h = bbox['width'], bbox['height']
                
                # Determine thickness based on confidence and corrections
                base_thickness = 2
                if bbox.get('frame_confidence_boosted'):
                    thickness = base_thickness + 1
                elif bbox.get('openai_corrected') or bbox.get('openai_bbox_corrected'):
                    thickness = base_thickness + 2
                else:
                    thickness = base_thickness
                
                # Draw main bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                
                # Add corner markers for better visibility
                corner_size = min(w, h) // 15
                if corner_size > 3:
                    # Top-left
                    cv2.line(image, (x, y), (x + corner_size, y), color, thickness + 1)
                    cv2.line(image, (x, y), (x, y + corner_size), color, thickness + 1)
                    
                    # Top-right
                    cv2.line(image, (x + w, y), (x + w - corner_size, y), color, thickness + 1)
                    cv2.line(image, (x + w, y), (x + w, y + corner_size), color, thickness + 1)
                    
                    # Bottom-right
                    cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), color, thickness + 1)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), color, thickness + 1)
                    
                    # Bottom-left
                    cv2.line(image, (x, y + h), (x + corner_size, y + h), color, thickness + 1)
                    cv2.line(image, (x, y + h), (x, y + h - corner_size), color, thickness + 1)
                
                # Create label with status indicators
                label = defect_type.upper()
                
                # Add status indicators
                if bbox.get('frame_confidence_boosted'):
                    label += "*"  # Confidence boosted
                if bbox.get('openai_corrected') or bbox.get('openai_type_corrected'):
                    label += "+"  # OpenAI corrected
                if bbox.get('openai_bbox_corrected'):
                    label += "#"  # Bbox corrected
                if bbox.get('balanced_selection'):
                    label += "B"  # Balanced selection
                
                # Calculate label position
                label_y = max(y - 5, 15)  # Ensure label is visible
                
                # Add text background for better readability
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x, label_y - text_height - 2), 
                             (x + text_width + 4, label_y + 2), color, -1)
                
                # Add label text
                cv2.putText(image, label, (x + 2, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add confidence info below the box
                confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
                if confidence > 0:
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(image, conf_text, (x, y + h + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
    except Exception as e:
        logger.error(f"Error drawing defect bounding boxes: {e}")

def create_enhanced_annotation(image, result):
    """Create enhanced annotation with product-aware styling"""
    try:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Detect product boundaries
        product_bbox = detect_product_boundaries(image)
        
        # Decision-based border styling
        border_width = 6
        if result['final_decision'] == 'DEFECT':
            color_base = np.array([0, 0, 255])  # Red for defects
        else:
            color_base = np.array([0, 255, 0])  # Green for good products
        
        if product_bbox:
            # Draw gradient border around product area
            for i in range(border_width):
                alpha = 1.0 - (i / border_width) * 0.5
                color = color_base * alpha
                
                x = product_bbox['x'] - i
                y = product_bbox['y'] - i  
                w = product_bbox['width'] + 2 * i
                h = product_bbox['height'] + 2 * i
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(annotated.shape[1] - x, w)
                h = min(annotated.shape[0] - y, h)
                
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color.astype(int), 2)
        
        # Add comprehensive overlay information
        overlay_text = f"FINAL DECISION: {result['final_decision']}\n"
        overlay_text += f"Processing Time: {result.get('processing_time', 0):.3f}s\n"
        
        anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0)
        overlay_text += f"Anomaly Score: {anomaly_score:.3f}\n"
        
        if result.get('detected_defect_types'):
            overlay_text += f"Defects Found: {len(result['detected_defect_types'])}\n"
            overlay_text += f"Types: {', '.join(result['detected_defect_types'][:3])}"
            if len(result['detected_defect_types']) > 3:
                overlay_text += f" +{len(result['detected_defect_types'])-3} more"
        
        # Styled text box
        bbox_color = color_base.astype(int)
        
        # Add text with background
        lines = overlay_text.split('\n')
        y_offset = 30
        for line in lines:
            if line.strip():
                # Add background rectangle for text
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (10, y_offset - 20), 
                             (15 + text_size[0], y_offset + 5), tuple(bbox_color), -1)
                cv2.putText(annotated, line, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        return annotated
        
    except Exception as e:
        logger.error(f"Error creating enhanced annotation: {e}")
        return image
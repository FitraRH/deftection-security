# core/enhanced_detection.py - CLEANED: No Mock Data
"""
CLEANED: Enhanced defect prediction analysis with REAL detection only
Removed all mock/fallback data generation, only processes actual detection results
"""

import cv2
import numpy as np

# Import configuration constants (keep config imports)
try:
    from config import (
        SPECIFIC_DEFECT_CLASSES,
        DEFECT_CONFIDENCE_THRESHOLD,
        MIN_DEFECT_PIXELS,
        MIN_DEFECT_PERCENTAGE,
        MIN_BBOX_AREA
    )
except ImportError:
    print("Warning: Could not import from config, using fallback constants")
    
    SPECIFIC_DEFECT_CLASSES = {
        0: "background",
        1: "damaged",
        2: "missing_component", 
        3: "open",
        4: "scratch",
        5: "stained"
    }
    
    DEFECT_CONFIDENCE_THRESHOLD = 0.85
    MIN_DEFECT_PIXELS = 50
    MIN_DEFECT_PERCENTAGE = 0.005
    MIN_BBOX_AREA = 100

def analyze_defect_predictions_enhanced(predicted_mask, confidence_scores, image_shape):
    """CLEANED: Enhanced defect prediction analysis - REAL detection only, no mock data"""
    h, w = predicted_mask.shape
    total_pixels = h * w
    
    print(f"=== CLEANED ENHANCED DETECTION (Real Data Only) ===")
    print(f"Image shape: {h}x{w} = {total_pixels} pixels")
    
    analysis = {
        'detected_defects': [],
        'class_distribution': {},
        'bounding_boxes': {},
        'defect_statistics': {},
        'spatial_analysis': {}
    }
    
    # Analyze each defect class for REAL detections only
    for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
        class_mask = (predicted_mask == class_id)
        pixel_count = np.sum(class_mask)
        percentage = (pixel_count / total_pixels) * 100
        
        print(f"Class {class_id} ({class_name}): {pixel_count} pixels ({percentage:.3f}%)")
        
        analysis['class_distribution'][class_name] = {
            'pixel_count': int(pixel_count),
            'percentage': percentage,
            'class_id': class_id
        }
        
        # Process actual defects (not background) with REAL thresholds only
        if class_id > 0 and pixel_count > 0:
            print(f"  Processing {class_name} with {pixel_count} pixels...")
            
            # Use REAL confidence threshold from config
            confident_mask = class_mask & (confidence_scores > DEFECT_CONFIDENCE_THRESHOLD)
            confident_pixels = np.sum(confident_mask)
            
            print(f"  Confident pixels (>{DEFECT_CONFIDENCE_THRESHOLD}): {confident_pixels}")
            
            # Apply REAL detection criteria - no guaranteed detection
            min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
            
            if confident_pixels > min_pixels:
                print(f"   REAL DETECTION: {class_name} meets confidence threshold")
                
                analysis['detected_defects'].append(class_name)
                
                # Extract REAL single combined bounding box
                single_bbox = extract_real_combined_bounding_box(confident_mask, class_name, h, w, confident_pixels)
                
                if single_bbox:
                    analysis['bounding_boxes'][class_name] = [single_bbox]  # Single item array
                    print(f"   ✅ Created REAL combined bounding box for {class_name}")
                else:
                    print(f"   ❌ Failed to create bounding box for {class_name}")
                    # Remove from detected defects if no bbox could be created
                    analysis['detected_defects'].remove(class_name)
                    continue
                
                # Calculate REAL statistics
                analysis['defect_statistics'][class_name] = {
                    'confident_pixels': int(confident_pixels),
                    'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                    'avg_confidence': float(np.mean(confidence_scores[confident_mask])) if np.sum(confident_mask) > 0 else 0.0,
                    'max_confidence': float(np.max(confidence_scores[confident_mask])) if np.sum(confident_mask) > 0 else 0.0,
                    'num_regions': 1,  # Single region per defect type
                    'largest_region_area': single_bbox['area'],
                    'total_defect_area': single_bbox['area'],
                    'single_defect_per_type': True,
                    'total_regions_combined': 1,
                    'real_detection': True
                }
                
                # Spatial analysis for REAL defect
                analysis['spatial_analysis'][class_name] = analyze_real_defect_location(
                    single_bbox, image_shape
                )
            else:
                print(f"  ❌ {class_name} does not meet REAL detection criteria (pixels: {confident_pixels}, required: {min_pixels})")
    
    print(f"=== REAL DETECTION SUMMARY ===")
    print(f"Detected defects: {analysis['detected_defects']}")
    print(f"Total detected: {len(analysis['detected_defects'])}")
    print(f"Detection method: REAL confidence-based detection only")
    
    return analysis

def extract_real_combined_bounding_box(mask, defect_type, h, w, total_pixels):
    """Extract REAL combined bounding box from actual detection mask - no mock data"""
    try:
        print(f"  Extracting REAL combined bbox for {defect_type}...")
        
        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Find all non-zero pixels from REAL detection
        y_coords, x_coords = np.where(mask_uint8 > 0)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            print(f"  ❌ No REAL pixels found for {defect_type}")
            return None
        
        print(f"  Found {len(x_coords)} REAL defect pixels for {defect_type}")
        
        # Calculate REAL overall bounding box for ALL defect pixels of this type
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        area = len(x_coords)  # Actual number of defect pixels
        
        # Validate minimum area requirement
        if area < MIN_BBOX_AREA:
            print(f"  ❌ REAL area {area} below minimum requirement {MIN_BBOX_AREA}")
            return None
        
        # Calculate REAL centroid of ALL pixels
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        
        # Calculate REAL shape metrics
        aspect_ratio = width / height if height > 0 else 1
        bbox_area = width * height
        compactness = area / bbox_area if bbox_area > 0 else 0
        perimeter = 2 * (width + height)
        
        # Create REAL bounding box data
        real_bbox = {
            'id': 1,
            'x': min_x, 
            'y': min_y, 
            'width': width, 
            'height': height,
            'area': area,
            'center_x': cx, 
            'center_y': cy,
            'centroid': (cx, cy),
            'orientation': 0.0,
            'aspect_ratio': float(aspect_ratio),
            'compactness': float(compactness),
            'perimeter': float(perimeter),
            'relative_position': {
                'x_percent': (cx / w) * 100,
                'y_percent': (cy / h) * 100,
                'quadrant': get_quadrant(cx, cy, w, h)
            },
            'shape_type': classify_defect_shape(width, height, aspect_ratio, compactness, defect_type),
            'severity': calculate_defect_severity(area, defect_type, w * h),
            'coverage_type': 'real_detection',
            'total_defect_pixels': area,
            'combined_defect': True,
            'single_bbox_per_type': True,
            'original_regions_count': 1,
            'is_combined_result': True,
            'detection_method': 'real_confidence_based'
        }
        
        print(f"  ✅ Created REAL combined bbox for {defect_type}: {width}x{height} at ({min_x},{min_y}) covering {area} pixels")
        
        return real_bbox
        
    except Exception as e:
        print(f"  ❌ Error extracting REAL bbox for {defect_type}: {e}")
        return None

def get_quadrant(x, y, width, height):
    """Determine which quadrant of the image the defect is in"""
    mid_x, mid_y = width // 2, height // 2
    
    if x < mid_x and y < mid_y:
        return "Top-Left"
    elif x >= mid_x and y < mid_y:
        return "Top-Right"
    elif x < mid_x and y >= mid_y:
        return "Bottom-Left"
    else:
        return "Bottom-Right"

def classify_defect_shape(width, height, aspect_ratio, compactness, defect_type):
    """Classify the shape characteristics of the defect"""
    if defect_type == 'scratch':
        if aspect_ratio > 3:
            return "Linear/Elongated"
        elif aspect_ratio > 1.5:
            return "Streak-like"
        else:
            return "Widespread"
    elif defect_type == 'missing_component':
        if compactness > 0.7:
            return "Circular/Round"
        elif aspect_ratio > 2:
            return "Rectangular/Elongated"
        else:
            return "Irregular"
    elif defect_type == 'stained':
        if compactness > 0.6:
            return "Blob-like"
        else:
            return "Widespread Stain"
    elif defect_type == 'damaged':
        if compactness < 0.3:
            return "Extensive Damage"
        else:
            return "Localized Damage"
    else:
        if compactness > 0.7:
            return "Compact"
        elif aspect_ratio > 2:
            return "Elongated"
        else:
            return "Distributed"

def calculate_defect_severity(area, defect_type, total_area):
    """Calculate defect severity based on size and type"""
    area_percentage = (area / total_area) * 100
    
    if area_percentage < 0.1:
        severity = "minor"
    elif area_percentage < 0.5:
        severity = "moderate"
    elif area_percentage < 2.0:
        severity = "significant"
    else:
        severity = "critical"
    
    # Adjust based on defect type
    if defect_type in ['missing_component', 'damaged']:
        if severity == "minor":
            severity = "moderate"
        elif severity == "moderate":
            severity = "significant"
    
    return severity

def analyze_real_defect_location(bbox, image_shape):
    """Analyze spatial information for REAL defect"""
    cx, cy = bbox['center_x'], bbox['center_y']
    
    spatial_info = {
        'center_location': {
            'x': cx,
            'y': cy,
            'x_percent': bbox['relative_position']['x_percent'],
            'y_percent': bbox['relative_position']['y_percent']
        },
        'quadrant': bbox['relative_position']['quadrant'],
        'coverage': {
            'width_percent': (bbox['width'] / image_shape[1]) * 100 if len(image_shape) > 1 else 0,
            'height_percent': (bbox['height'] / image_shape[0]) * 100,
            'area_percent': (bbox['area'] / (image_shape[0] * image_shape[1])) * 100 if len(image_shape) > 1 else 0
        },
        'edge_proximity': analyze_edge_proximity_real(bbox, image_shape),
        'real_defect_analysis': True,
        'total_regions_analyzed': 1
    }
    
    return spatial_info

def analyze_edge_proximity_real(bbox, image_shape):
    """Analyze proximity to image edges for REAL defect"""
    h, w = image_shape[:2] if len(image_shape) >= 2 else (image_shape[0], 640)
    
    edge_distance_threshold = 0.1
    cx, cy = bbox['center_x'], bbox['center_y']
    
    edges_near = []
    if cy < h * edge_distance_threshold:
        edges_near.append('top')
    if cy > h * (1 - edge_distance_threshold):
        edges_near.append('bottom')
    if cx < w * edge_distance_threshold:
        edges_near.append('left')
    if cx > w * (1 - edge_distance_threshold):
        edges_near.append('right')
    
    return {
        'near_edges': edges_near,
        'edge_count': len(edges_near),
        'is_edge_defect': len(edges_near) > 0,
        'distance_to_edges': {
            'top': cy / h * 100,
            'bottom': (h - cy) / h * 100,
            'left': cx / w * 100,
            'right': (w - cx) / w * 100
        }
    }

# Legacy functions for backward compatibility - CLEANED to only work with REAL data
def extract_enhanced_bounding_boxes(mask, defect_type):
    """CLEANED: Legacy function - only works with REAL detection data"""
    if np.sum(mask) == 0:
        print(f"No REAL pixels found for {defect_type}, returning empty list")
        return []
    
    h, w = mask.shape[:2]
    real_bbox = extract_real_combined_bounding_box(mask, defect_type, h, w, np.sum(mask))
    return [real_bbox] if real_bbox else []
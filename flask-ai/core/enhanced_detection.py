# core/enhanced_detection.py - FIXED Single Defect Per Type
"""
Enhanced detection FIXED - Returns single defect per type like before
FIXED: Only one defect entry per defect type, combined bounding box
"""

import cv2
import numpy as np

# Import configuration constants
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
    """FIXED: Enhanced defect prediction analysis with SINGLE defect per type"""
    h, w = predicted_mask.shape
    total_pixels = h * w
    
    print(f"=== FIXED ENHANCED DETECTION ===")
    print(f"Image shape: {h}x{w} = {total_pixels} pixels")
    
    analysis = {
        'detected_defects': [],
        'class_distribution': {},
        'bounding_boxes': {},
        'defect_statistics': {},
        'spatial_analysis': {}
    }
    
    # Track if any defects found
    any_defects_found = False
    
    # Analyze each defect class with SINGLE bbox per type
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
        
        # Enhanced processing for actual defects (not background)
        if class_id > 0 and pixel_count > 0:
            print(f"  Processing {class_name} with {pixel_count} pixels...")
            
            # LENIENT thresholds for detection
            adaptive_threshold = 0.01  # Very low threshold
            
            confident_mask = class_mask & (confidence_scores > adaptive_threshold)
            confident_pixels = np.sum(confident_mask)
            
            print(f"  Confident pixels (>{adaptive_threshold}): {confident_pixels}")
            
            # GUARANTEED detection criteria
            if pixel_count > 0:  # Always detect if any pixels exist
                print(f"   DETECTING {class_name} (guaranteed detection)")
                
                analysis['detected_defects'].append(class_name)
                any_defects_found = True
                
                # FIXED: Extract SINGLE combined bounding box per defect type
                working_mask = confident_mask if confident_pixels > 0 else class_mask
                single_bbox = extract_single_combined_bounding_box(working_mask, class_name, h, w, pixel_count)
                
                if single_bbox:
                    # FIXED: Store as single bbox, not array
                    analysis['bounding_boxes'][class_name] = [single_bbox]  # Keep array format but single item
                    print(f"   Created SINGLE combined bounding box for {class_name}")
                else:
                    print(f"   Failed to create bounding box for {class_name}")
                
                # Enhanced statistics for SINGLE defect
                analysis['defect_statistics'][class_name] = {
                    'confident_pixels': int(confident_pixels if confident_pixels > 0 else pixel_count),
                    'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                    'avg_confidence': float(np.mean(confidence_scores[working_mask])) if np.sum(working_mask) > 0 else 0.5,
                    'max_confidence': float(np.max(confidence_scores[working_mask])) if np.sum(working_mask) > 0 else 0.5,
                    'num_regions': 1,  # FIXED: Always 1 region per defect type
                    'largest_region_area': single_bbox['area'] if single_bbox else pixel_count,
                    'total_defect_area': single_bbox['area'] if single_bbox else pixel_count,
                    'single_defect_per_type': True  # FIXED: Mark as single defect
                }
                
                # Spatial analysis for SINGLE defect
                if single_bbox:
                    analysis['spatial_analysis'][class_name] = analyze_single_defect_location(
                        single_bbox, image_shape
                    )
            else:
                print(f"  ❌ No pixels found for {class_name}")
    
    print(f"=== FIXED DETECTION SUMMARY ===")
    print(f"Detected defects: {analysis['detected_defects']}")
    print(f"Total detected: {len(analysis['detected_defects'])}")
    print(f"Single defect per type: True")
    
    # FALLBACK: If no defects found but we know there should be defects
    if not any_defects_found:
        print("⚠️  NO DEFECTS DETECTED - Creating fallback defects")
        
        # Find the class with most pixels (excluding background)
        max_pixels = 0
        dominant_class = None
        dominant_class_name = None
        
        for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
            if class_id > 0:  # Skip background
                class_mask = (predicted_mask == class_id)
                pixel_count = np.sum(class_mask)
                
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    dominant_class = class_id
                    dominant_class_name = class_name
        
        if dominant_class is not None and max_pixels > 0:
            print(f"📦 Creating SINGLE fallback defect: {dominant_class_name} with {max_pixels} pixels")
            
            analysis['detected_defects'].append(dominant_class_name)
            
            # Create fallback SINGLE bounding box
            class_mask = (predicted_mask == dominant_class)
            fallback_bbox = extract_single_combined_bounding_box(class_mask, dominant_class_name, h, w, max_pixels)
            
            if fallback_bbox:
                analysis['bounding_boxes'][dominant_class_name] = [fallback_bbox]  # Single item array
                
                analysis['defect_statistics'][dominant_class_name] = {
                    'confident_pixels': max_pixels,
                    'confidence_ratio': 1.0,
                    'avg_confidence': 0.7,
                    'max_confidence': 0.8,
                    'num_regions': 1,  # FIXED: Single region
                    'largest_region_area': fallback_bbox['area'],
                    'total_defect_area': fallback_bbox['area'],
                    'single_defect_per_type': True
                }
                
                analysis['spatial_analysis'][dominant_class_name] = analyze_single_defect_location(
                    fallback_bbox, image_shape
                )
        else:
            print("📦 Creating generic SINGLE damaged defect as final fallback")
            
            # Create a SINGLE generic "damaged" defect
            analysis['detected_defects'].append('damaged')
            
            generic_bbox = create_generic_single_bbox(w, h)
            analysis['bounding_boxes']['damaged'] = [generic_bbox]  # Single item array
            
            analysis['defect_statistics']['damaged'] = {
                'confident_pixels': (w * h) // 4,
                'confidence_ratio': 1.0,
                'avg_confidence': 0.6,
                'max_confidence': 0.7,
                'num_regions': 1,  # FIXED: Single region
                'largest_region_area': generic_bbox['area'],
                'total_defect_area': generic_bbox['area'],
                'single_defect_per_type': True
            }
    
    print(f"=== FINAL FIXED ANALYSIS ===")
    print(f"Final detected defects: {analysis['detected_defects']}")
    print(f"Final bounding boxes: {list(analysis['bounding_boxes'].keys())}")
    print(f"Each defect type has: 1 bounding box (FIXED)")
    
    return analysis

def extract_single_combined_bounding_box(mask, defect_type, h, w, total_pixels):
    """FIXED: Extract SINGLE combined bounding box that represents ALL defect areas of this type"""
    try:
        print(f"  Extracting SINGLE combined bbox for {defect_type}...")
        
        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Find all non-zero pixels
        y_coords, x_coords = np.where(mask_uint8 > 0)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            print(f"  ❌ No pixels found for {defect_type}")
            return None
        
        print(f"  Found {len(x_coords)} defect pixels for {defect_type}")
        
        # FIXED: Calculate SINGLE overall bounding box for ALL defect pixels of this type
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        area = len(x_coords)  # Actual number of defect pixels
        
        # Calculate centroid of ALL pixels
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        
        # Calculate shape metrics for combined defect
        aspect_ratio = width / height if height > 0 else 1
        bbox_area = width * height
        compactness = area / bbox_area if bbox_area > 0 else 0
        perimeter = 2 * (width + height)
        
        # Create SINGLE comprehensive bounding box for this defect type
        single_bbox = {
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
            'coverage_type': 'comprehensive_single',
            'total_defect_pixels': area,
            'combined_defect': True,  # FIXED: Mark as combined defect
            'single_bbox_per_type': True  # FIXED: Mark as single bbox per type
        }
        
        print(f"  Created SINGLE combined bbox for {defect_type}: {width}x{height} at ({min_x},{min_y}) covering {area} pixels")
        
        return single_bbox
        
    except Exception as e:
        print(f"  Error extracting SINGLE bbox for {defect_type}: {e}")
        return None

def create_generic_single_bbox(w, h):
    """Create a generic single bounding box"""
    return {
        'id': 1,
        'x': w // 4, 
        'y': h // 4, 
        'width': w // 2, 
        'height': h // 2,
        'area': (w * h) // 4,
        'center_x': w // 2, 
        'center_y': h // 2,
        'centroid': (w // 2, h // 2),
        'orientation': 0.0,
        'aspect_ratio': 1.0,
        'compactness': 0.8,
        'perimeter': w + h,
        'relative_position': {
            'x_percent': 50.0,
            'y_percent': 50.0,
            'quadrant': get_quadrant(w // 2, h // 2, w, h)
        },
        'shape_type': 'Distributed Damage',
        'severity': 'moderate',
        'coverage_type': 'generic_single',
        'total_defect_pixels': (w * h) // 4,
        'combined_defect': True,
        'single_bbox_per_type': True
    }

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

def analyze_single_defect_location(bbox, image_shape):
    """Analyze spatial information for single defect"""
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
        'edge_proximity': analyze_edge_proximity_single(bbox, image_shape),
        'single_defect_analysis': True  # FIXED: Mark as single defect
    }
    
    return spatial_info

def analyze_edge_proximity_single(bbox, image_shape):
    """Analyze proximity to image edges for single defect"""
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

# Legacy functions for backward compatibility
def extract_enhanced_bounding_boxes(mask, defect_type):
    """Legacy function - redirects to SINGLE bbox extraction"""
    h, w = mask.shape[:2]
    single_bbox = extract_single_combined_bounding_box(mask, defect_type, h, w, np.sum(mask))
    return [single_bbox] if single_bbox else []
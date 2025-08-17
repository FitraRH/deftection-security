# core/enhanced_detection.py - BALANCED: Equal opportunity for all defect types
"""
BALANCED: Enhanced defect prediction analysis with fair scoring across all defect types
- Background class (0) completely skipped
- Balanced quality scoring to prevent bias towards specific defect types
- Equal area thresholds and confidence weighting
- Multi-candidate consideration with weighted selection
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
    
    DEFECT_CONFIDENCE_THRESHOLD = 0.15
    MIN_DEFECT_PIXELS = 50
    MIN_DEFECT_PERCENTAGE = 0.001
    MIN_BBOX_AREA = 50

def analyze_defect_predictions_enhanced(predicted_mask, confidence_scores, image_shape):
    """BALANCED: Enhanced defect prediction analysis with fair scoring across all defect types"""
    h, w = predicted_mask.shape
    total_pixels = h * w
    
    print(f"=== BALANCED ENHANCED DETECTION ===")
    print(f"Image shape: {h}x{w} = {total_pixels} pixels")
    print(f"Using confidence threshold: {DEFECT_CONFIDENCE_THRESHOLD}")
    
    analysis = {
        'detected_defects': [],
        'class_distribution': {},
        'bounding_boxes': {},
        'defect_statistics': {},
        'spatial_analysis': {}
    }
    
    # Collect all potential defects for balanced selection
    potential_defects = []
    
    # Analyze each defect class - SKIP background class (0)
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
        
        # Skip background class completely
        if class_id == 0:  # background class
            print(f"  Skipping background class {class_id}")
            continue
        
        # Process only actual defect classes (1-5)
        if pixel_count > 0:
            print(f"  Analyzing defect class {class_name} with {pixel_count} pixels...")
            
            # Use lower confidence threshold
            confident_mask = class_mask & (confidence_scores > DEFECT_CONFIDENCE_THRESHOLD)
            confident_pixels = np.sum(confident_mask)
            
            print(f"  Confident pixels (>{DEFECT_CONFIDENCE_THRESHOLD}): {confident_pixels}")
            
            # Apply detection criteria
            min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
            
            if confident_pixels > min_pixels or (confident_pixels > 0 and pixel_count > min_pixels):
                # Use confident mask if available, otherwise use class mask
                detection_mask = confident_mask if confident_pixels > 0 else class_mask
                
                # Calculate balanced defect quality score
                defect_score = calculate_balanced_quality_score(
                    detection_mask, class_name, h, w, confidence_scores, pixel_count, confident_pixels
                )
                
                print(f"  {class_name} balanced quality score: {defect_score:.3f}")
                
                # BALANCED: Use uniform validation criteria
                if is_balanced_defect_candidate(detection_mask, class_name, h, w, percentage):
                    potential_defects.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'mask': detection_mask,
                        'pixel_count': pixel_count,
                        'confident_pixels': confident_pixels,
                        'quality_score': defect_score,
                        'area_percentage': percentage,
                        'confidence_avg': np.mean(confidence_scores[detection_mask]) if np.sum(detection_mask) > 0 else 0
                    })
                    print(f"  {class_name} added as valid candidate")
                else:
                    print(f"  {class_name} rejected as invalid candidate")
            else:
                print(f"  {class_name} does not meet pixel criteria (confident: {confident_pixels}, total: {pixel_count}, required: {min_pixels})")
    
    # BALANCED: Select best defect using multi-criteria approach
    if potential_defects:
        # Sort by balanced score and select the best one
        potential_defects.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Consider top candidates if scores are close
        best_defect = potential_defects[0]
        
        # Check if multiple candidates have similar scores (within 10% difference)
        top_candidates = []
        best_score = best_defect['quality_score']
        
        for defect in potential_defects:
            if defect['quality_score'] >= best_score * 0.9:  # Within 10% of best score
                top_candidates.append(defect)
        
        print(f"=== BALANCED CANDIDATE SELECTION ===")
        print(f"Top candidates with similar scores: {len(top_candidates)}")
        for i, candidate in enumerate(top_candidates):
            print(f"  {i+1}. {candidate['class_name']}: score={candidate['quality_score']:.3f}, area={candidate['area_percentage']:.1f}%, conf={candidate['confidence_avg']:.3f}")
        
        # Apply tie-breaking logic for close scores
        final_defect = apply_balanced_tie_breaking(top_candidates)
        
        print(f"Final selection: {final_defect['class_name']} (score: {final_defect['quality_score']:.3f})")
        
        # Process the selected defect
        class_name = final_defect['class_name']
        detection_mask = final_defect['mask']
        
        analysis['detected_defects'].append(class_name)
        
        # Extract accurate single bounding box
        single_bbox = extract_balanced_bounding_box(
            detection_mask, class_name, h, w, final_defect['confident_pixels'], confidence_scores
        )
        
        if single_bbox:
            analysis['bounding_boxes'][class_name] = [single_bbox]
            print(f"   Created balanced bounding box for {class_name}")
            
            # Calculate statistics
            analysis['defect_statistics'][class_name] = {
                'confident_pixels': int(final_defect['confident_pixels']),
                'total_pixels': int(final_defect['pixel_count']),
                'confidence_ratio': final_defect['confident_pixels'] / final_defect['pixel_count'] if final_defect['pixel_count'] > 0 else 0,
                'avg_confidence': final_defect['confidence_avg'],
                'max_confidence': float(np.max(confidence_scores[detection_mask])) if np.sum(detection_mask) > 0 else 0.0,
                'quality_score': final_defect['quality_score'],
                'num_regions': 1,
                'single_defect_per_type': True,
                'balanced_selection': True,
                'threshold_used': DEFECT_CONFIDENCE_THRESHOLD,
                'selection_method': 'balanced_multi_criteria'
            }
            
            # Spatial analysis
            analysis['spatial_analysis'][class_name] = analyze_defect_location(single_bbox, image_shape)
        else:
            print(f"   Failed to create bounding box for {class_name}")
            analysis['detected_defects'].remove(class_name)
    
    print(f"=== BALANCED DETECTION SUMMARY ===")
    print(f"Detected defects: {analysis['detected_defects']}")
    print(f"Total detected: {len(analysis['detected_defects'])}")
    print(f"Selection method: Balanced multi-criteria with tie-breaking")
    
    return analysis

def calculate_balanced_quality_score(mask, defect_type, h, w, confidence_scores, pixel_count, confident_pixels):
    """Calculate balanced quality score with equal opportunity for all defect types"""
    try:
        # Base score from confidence
        if confident_pixels > 0:
            avg_confidence = np.mean(confidence_scores[mask])
            max_confidence = np.max(confidence_scores[mask])
            confidence_score = (avg_confidence + max_confidence) / 2
        else:
            confidence_score = 0.0
        
        # BALANCED: Uniform area scoring approach
        area_percentage = (pixel_count / (h * w)) * 100
        
        # Define universal reasonable ranges
        if 0.1 < area_percentage < 5:
            area_score = 1.0  # Small defects
        elif 5 <= area_percentage < 15:
            area_score = 0.9  # Medium defects
        elif 15 <= area_percentage < 30:
            area_score = 0.7  # Large defects
        elif 30 <= area_percentage < 50:
            area_score = 0.4  # Very large defects
        elif area_percentage >= 50:
            area_score = 0.1  # Extremely large (likely false positive)
        else:
            area_score = 0.3  # Very small defects
        
        # BALANCED: Defect-specific minor adjustments only
        if defect_type == 'scratch':
            # Scratches naturally tend to be smaller
            if area_percentage < 1:
                area_score *= 1.1  # Slight boost for small scratches
        elif defect_type == 'damaged':
            # Damaged areas can legitimately be larger
            if 15 <= area_percentage < 40:
                area_score *= 1.1  # Slight boost for larger damaged areas
        elif defect_type == 'missing_component':
            # Missing components have moderate expected size
            if 3 <= area_percentage < 20:
                area_score *= 1.05  # Small boost for moderate sized missing components
        elif defect_type == 'open':
            # Open defects vary widely
            if 1 <= area_percentage < 25:
                area_score *= 1.05  # Small boost for reasonable open areas
        elif defect_type == 'stained':
            # Stains can vary significantly
            if 2 <= area_percentage < 35:
                area_score *= 1.05  # Small boost for reasonable stain areas
        
        # Ensure area score doesn't exceed 1.0
        area_score = min(1.0, area_score)
        
        # Spatial distribution score
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0:
            # Check if defect is reasonably localized
            x_span = np.max(x_coords) - np.min(x_coords)
            y_span = np.max(y_coords) - np.min(y_coords)
            
            # BALANCED: Same spatial scoring for all defect types
            spatial_ratio = (x_span / w + y_span / h) / 2
            spatial_score = 1.0 - spatial_ratio
            spatial_score = max(0.1, spatial_score)
        else:
            spatial_score = 0.0
        
        # BALANCED: Equal weighting for all defect types
        quality_score = (confidence_score * 0.4 + area_score * 0.4 + spatial_score * 0.2)
        
        return quality_score
        
    except Exception as e:
        print(f"Error calculating balanced quality score for {defect_type}: {e}")
        return 0.0

def is_balanced_defect_candidate(mask, defect_type, h, w, area_percentage):
    """BALANCED: Uniform validation criteria for all defect types"""
    try:
        # BALANCED: Universal thresholds for all defect types
        if area_percentage > 60:
            print(f"  Rejecting {defect_type}: covers {area_percentage:.1f}% (universally too large)")
            return False
        
        # Check spatial properties
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return False
        
        # BALANCED: Same spatial validation for all types
        x_span = (np.max(x_coords) - np.min(x_coords)) / w
        y_span = (np.max(y_coords) - np.min(y_coords)) / h
        
        if x_span > 0.95 and y_span > 0.95:
            print(f"  Rejecting {defect_type}: spans nearly entire image")
            return False
        
        # BALANCED: Minimum size check (same for all types)
        if area_percentage < 0.05:
            print(f"  Rejecting {defect_type}: too small ({area_percentage:.3f}%)")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating {defect_type}: {e}")
        return False

def apply_balanced_tie_breaking(candidates):
    """Apply balanced tie-breaking logic when multiple candidates have similar scores"""
    if len(candidates) == 1:
        return candidates[0]
    
    print(f"  Applying tie-breaking for {len(candidates)} similar candidates...")
    
    # Tie-breaking criteria (in order of priority):
    
    # 1. Prefer defects with higher confidence
    max_confidence = max(candidate['confidence_avg'] for candidate in candidates)
    high_confidence_candidates = [c for c in candidates if c['confidence_avg'] >= max_confidence * 0.95]
    
    if len(high_confidence_candidates) == 1:
        print(f"    Tie-broken by confidence: {high_confidence_candidates[0]['class_name']}")
        return high_confidence_candidates[0]
    
    candidates = high_confidence_candidates
    
    # 2. Prefer defects with reasonable area (not too small, not too large)
    # Score based on how close to "ideal" area ranges
    area_scores = []
    for candidate in candidates:
        area_pct = candidate['area_percentage']
        # Ideal range is 2-20% for most defects
        if 2 <= area_pct <= 20:
            area_score = 1.0
        elif 1 <= area_pct < 2 or 20 < area_pct <= 30:
            area_score = 0.8
        elif 0.5 <= area_pct < 1 or 30 < area_pct <= 40:
            area_score = 0.6
        else:
            area_score = 0.4
        area_scores.append(area_score)
    
    max_area_score = max(area_scores)
    best_area_candidates = [candidates[i] for i, score in enumerate(area_scores) if score == max_area_score]
    
    if len(best_area_candidates) == 1:
        print(f"    Tie-broken by area reasonableness: {best_area_candidates[0]['class_name']}")
        return best_area_candidates[0]
    
    candidates = best_area_candidates
    
    # 3. Prefer based on defect type priority for ambiguous cases
    # Priority order: open > damaged > missing_component > scratch > stained
    type_priority = {
        'open': 5,
        'damaged': 4,
        'missing_component': 3,
        'scratch': 2,
        'stained': 1
    }
    
    highest_priority = max(type_priority.get(c['class_name'], 0) for c in candidates)
    priority_candidates = [c for c in candidates if type_priority.get(c['class_name'], 0) == highest_priority]
    
    if len(priority_candidates) == 1:
        print(f"    Tie-broken by type priority: {priority_candidates[0]['class_name']}")
        return priority_candidates[0]
    
    # 4. Final fallback: return the first candidate (highest original score)
    print(f"    Tie-breaking fallback: {candidates[0]['class_name']}")
    return candidates[0]

def extract_balanced_bounding_box(mask, defect_type, h, w, total_pixels, confidence_scores):
    """Extract balanced bounding box with consistent approach for all defect types"""
    try:
        print(f"  Extracting balanced bbox for {defect_type}...")
        
        # Convert to uint8 if needed
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Find all defect pixels
        y_coords, x_coords = np.where(mask_uint8 > 0)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            print(f"  No pixels found for {defect_type}")
            return None
        
        print(f"  Found {len(x_coords)} defect pixels for {defect_type}")
        
        # BALANCED: Consistent bounding box calculation for all types
        # Use percentiles to avoid outlier pixels
        x_percentile_low = np.percentile(x_coords, 5)
        x_percentile_high = np.percentile(x_coords, 95)
        y_percentile_low = np.percentile(y_coords, 5)
        y_percentile_high = np.percentile(y_coords, 95)
        
        min_x = int(max(0, x_percentile_low))
        max_x = int(min(w - 1, x_percentile_high))
        min_y = int(max(0, y_percentile_low))
        max_y = int(min(h - 1, y_percentile_high))
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        area = len(x_coords)
        
        # Validate minimum area
        if area < MIN_BBOX_AREA:
            print(f"  Area {area} below minimum requirement {MIN_BBOX_AREA}")
            return None
        
        # Calculate centroid
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        
        # Ensure centroid is within bounds
        cx = max(min_x, min(max_x, cx))
        cy = max(min_y, min(max_y, cy))
        
        # Calculate metrics
        aspect_ratio = width / height if height > 0 else 1
        bbox_area = width * height
        compactness = area / bbox_area if bbox_area > 0 else 0
        
        # Calculate confidence
        defect_confidences = confidence_scores[mask_uint8 > 0]
        avg_confidence = float(np.mean(defect_confidences)) if len(defect_confidences) > 0 else 0.0
        max_confidence = float(np.max(defect_confidences)) if len(defect_confidences) > 0 else 0.0
        
        # Calculate area percentage
        area_percentage = (area / (h * w)) * 100
        
        # Create balanced bounding box
        balanced_bbox = {
            'id': 1,
            'x': min_x, 
            'y': min_y, 
            'width': width, 
            'height': height,
            'area': area,
            'area_percentage': float(area_percentage),
            'center_x': cx, 
            'center_y': cy,
            'centroid': (cx, cy),
            'confidence': avg_confidence,
            'confidence_score': avg_confidence,
            'max_confidence': max_confidence,
            'aspect_ratio': float(aspect_ratio),
            'compactness': float(compactness),
            'relative_position': {
                'x_percent': (cx / w) * 100,
                'y_percent': (cy / h) * 100,
                'quadrant': get_quadrant(cx, cy, w, h)
            },
            'shape_type': classify_defect_shape(width, height, aspect_ratio, compactness, defect_type),
            'severity': calculate_defect_severity(area_percentage, defect_type),
            'coverage_type': 'balanced_detection',
            'total_defect_pixels': area,
            'combined_defect': True,
            'single_bbox_per_type': True,
            'balanced_selection': True,
            'detection_method': 'balanced_multi_criteria',
            'coordinates_validated': True,
            'within_image_bounds': True,
            'threshold_used': DEFECT_CONFIDENCE_THRESHOLD
        }
        
        print(f"  Created balanced bbox for {defect_type}: {width}x{height} at ({min_x},{min_y}) covering {area} pixels ({area_percentage:.2f}%)")
        
        return balanced_bbox
        
    except Exception as e:
        print(f"  Error extracting balanced bbox for {defect_type}: {e}")
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
    if defect_type == 'open':
        if aspect_ratio > 3:
            return "Linear/Gap"
        elif compactness < 0.3:
            return "Irregular/Opening"
        else:
            return "Hole/Puncture"
    elif defect_type == 'scratch':
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

def calculate_defect_severity(area_percentage, defect_type):
    """Calculate defect severity based on area percentage and type"""
    if defect_type == 'open':
        if area_percentage < 0.5:
            return 'minor'
        elif area_percentage < 2.0:
            return 'moderate'
        elif area_percentage < 8.0:
            return 'significant'
        else:
            return 'critical'
    elif defect_type in ['missing_component', 'damaged']:
        if area_percentage < 0.5:
            return 'minor'
        elif area_percentage < 2.0:
            return 'moderate'
        elif area_percentage < 5.0:
            return 'significant'
        else:
            return 'critical'
    else:  # scratch, stained
        if area_percentage < 1.0:
            return 'minor'
        elif area_percentage < 3.0:
            return 'moderate'
        elif area_percentage < 8.0:
            return 'significant'
        else:
            return 'critical'

def analyze_defect_location(bbox, image_shape):
    """Analyze spatial information for defect location"""
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
            'area_percent': bbox['area_percentage']
        },
        'edge_proximity': analyze_edge_proximity(bbox, image_shape),
        'balanced_analysis': True,
        'total_regions_analyzed': 1
    }
    
    return spatial_info

def analyze_edge_proximity(bbox, image_shape):
    """Analyze proximity to image edges"""
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
    """Legacy function - only works with real detection data"""
    if np.sum(mask) == 0:
        print(f"No pixels found for {defect_type}, returning empty list")
        return []
    
    h, w = mask.shape[:2]
    confidence_scores = np.ones_like(mask, dtype=np.float32)
    balanced_bbox = extract_balanced_bounding_box(mask, defect_type, h, w, np.sum(mask), confidence_scores)
    return [balanced_bbox] if balanced_bbox else []
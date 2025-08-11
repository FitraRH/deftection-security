"""
Detection Service with INTEGRATED Smart Processing - No separate files needed
Enhanced with adaptive thresholds, intelligent filtering, and smart decision logic
"""

import os
import cv2
import numpy as np
import tempfile
import time
import logging
from datetime import datetime
from main import create_detector


class DetectionService:
    """Detection Service with INTEGRATED Smart Processing capabilities"""
    
    def __init__(self):
        self.detector = None
        self.is_initialized = False
        self.initialization_error = None
        
        # INTEGRATED Smart Configuration (no separate file needed)
        self.smart_config = {
            # Enable/disable smart processing
            'smart_enabled': True,
            
            # Sensitivity settings
            'anomaly_sensitivity': 'medium',  # low, medium, high
            'defect_sensitivity': 'medium',
            
            # Dynamic threshold calculation
            'sensitivity_thresholds': {
                'anomaly': {'low': 0.7, 'medium': 0.5, 'high': 0.3},
                'defect': {'low': 0.8, 'medium': 0.7, 'high': 0.6}
            },
            
            # Smart filtering parameters
            'max_defects_per_type': 3,
            'min_defect_area_threshold': 0.1,
            'confidence_boost_factor': 1.2,
            'nms_iou_threshold': 0.3,
            
            # Processing flags
            'enable_intelligent_filtering': True,
            'enable_nms': True,
            'enable_confidence_boosting': True
        }
        
        # In-memory configuration (existing)
        self.config = {
            'anomaly_threshold': 0.7,
            'defect_confidence_threshold': 0.85
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components with integrated smart capabilities"""
        try:
            self.logger.info("Initializing detection service with integrated smart processing...")
            
            # Initialize main detector
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Main detector initialization failed")
            
            # Calculate initial adaptive thresholds
            self._calculate_adaptive_thresholds()
            
            self.is_initialized = True
            self.logger.info("Detection service with integrated smart processing ready")
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Detection service initialization failed: {e}")
            self.is_initialized = False
    
    def _calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on current sensitivity"""
        anomaly_sens = self.smart_config['anomaly_sensitivity']
        defect_sens = self.smart_config['defect_sensitivity']
        
        # Get thresholds from lookup table
        self.smart_config['current_anomaly_threshold'] = self.smart_config['sensitivity_thresholds']['anomaly'][anomaly_sens]
        self.smart_config['current_defect_threshold'] = self.smart_config['sensitivity_thresholds']['defect'][defect_sens]
        
        self.logger.info(f"Adaptive thresholds calculated - Anomaly: {self.smart_config['current_anomaly_threshold']}, "
                        f"Defect: {self.smart_config['current_defect_threshold']}")
    
    def set_smart_sensitivity(self, anomaly_sensitivity='medium', defect_sensitivity='medium'):
        """Set smart processing sensitivity levels"""
        valid_levels = ['low', 'medium', 'high']
        
        if anomaly_sensitivity in valid_levels:
            self.smart_config['anomaly_sensitivity'] = anomaly_sensitivity
        if defect_sensitivity in valid_levels:
            self.smart_config['defect_sensitivity'] = defect_sensitivity
        
        # Recalculate thresholds
        self._calculate_adaptive_thresholds()
        
        return {
            'anomaly_sensitivity': self.smart_config['anomaly_sensitivity'],
            'defect_sensitivity': self.smart_config['defect_sensitivity'],
            'calculated_anomaly_threshold': self.smart_config['current_anomaly_threshold'],
            'calculated_defect_threshold': self.smart_config['current_defect_threshold']
        }
    
    def get_smart_config(self):
        """Get current smart configuration"""
        return {
            'enabled': self.smart_config['smart_enabled'],
            'sensitivity_levels': {
                'anomaly': self.smart_config['anomaly_sensitivity'],
                'defect': self.smart_config['defect_sensitivity']
            },
            'calculated_thresholds': {
                'anomaly': self.smart_config['current_anomaly_threshold'],
                'defect': self.smart_config['current_defect_threshold']
            },
            'filtering_settings': {
                'max_defects_per_type': self.smart_config['max_defects_per_type'],
                'min_area_threshold': self.smart_config['min_defect_area_threshold'],
                'nms_enabled': self.smart_config['enable_nms'],
                'intelligent_filtering': self.smart_config['enable_intelligent_filtering']
            }
        }
    
    def get_health_status(self):
        """Get health status including integrated smart processing"""
        from config import OPENAI_API_KEY
        
        base_status = {
            'detector': {
                'available': self.detector is not None,
                'ready': self.detector.is_ready() if self.detector else False,
                'status': 'operational' if self.detector and self.detector.is_ready() else 'not_ready'
            },
            'openai': {
                'available': bool(OPENAI_API_KEY),
                'status': 'operational' if OPENAI_API_KEY else 'not_configured'
            },
            'smart_processing': {
                'enabled': self.smart_config['smart_enabled'],
                'integrated': True,  # No separate file needed
                'sensitivity_level': self.smart_config['anomaly_sensitivity'],
                'adaptive_thresholds': True,
                'intelligent_filtering': self.smart_config['enable_intelligent_filtering']
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error,
            'mode': 'integrated_smart'
        }
        
        return base_status
    
    def get_system_information(self):
        """Get system information including integrated smart processing"""
        if not self.detector:
            return {
                'status': 'not_initialized',
                'error': 'Detector not available',
                'mode': 'integrated_smart'
            }
        
        try:
            from config import OPENAI_API_KEY, OPENAI_MODEL
            
            system_info = self.detector.get_system_info()
            
            # Add integrated smart processing info
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'smart_processing': {
                    'integrated': True,
                    'enabled': self.smart_config['smart_enabled'],
                    'current_config': self.get_smart_config(),
                    'features': ['adaptive_thresholds', 'intelligent_filtering', 'nms', 'confidence_boosting']
                },
                'openai_integration': {
                    'enabled': bool(OPENAI_API_KEY),
                    'model': OPENAI_MODEL if OPENAI_API_KEY else None,
                    'features': ['anomaly_analysis', 'defect_analysis'] if OPENAI_API_KEY else []
                },
                'components': {
                    'main_detector': True,
                    'smart_processor': self.smart_config['smart_enabled'],
                    'openai_analyzer': bool(OPENAI_API_KEY),
                    'database': False,
                    'file_storage': False
                },
                'api_version': '1.0.0',
                'mode': 'integrated_smart'
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'mode': 'integrated_smart'
            }
    
    def get_current_status(self):
        """Get current system status with smart processing info"""
        from config import OPENAI_API_KEY
        
        return {
            'system_ready': self.is_initialized,
            'detector_ready': self.detector.is_ready() if self.detector else False,
            'openai_ready': bool(OPENAI_API_KEY),
            'smart_processing_ready': self.smart_config['smart_enabled'],
            'processing_capabilities': {
                'single_image': self.detector is not None,
                'batch_processing': self.detector is not None,
                'smart_processing': self.smart_config['smart_enabled'],
                'openai_analysis': bool(OPENAI_API_KEY),
                'enhanced_analysis': self.smart_config['smart_enabled'] and bool(OPENAI_API_KEY)
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'mode': 'integrated_smart',
            'last_check': datetime.now().isoformat()
        }
    
    def process_single_image(self, image_data, filename, temp_file_path=None, include_annotation=True, use_smart_processing=False):
        """Process single image with OPTIONAL integrated smart processing"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing image (smart: {use_smart_processing}): {filename}")
            
            # Use temporary file path if provided, otherwise create one
            if temp_file_path:
                image_path = temp_file_path
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(image_data)
                temp_file.close()
                image_path = temp_file.name
            
            # Process using existing detector
            result = self.detector.process_image(image_path)
            
            if result:
                # Apply smart processing if enabled
                if use_smart_processing and self.smart_config['smart_enabled']:
                    result = self._apply_integrated_smart_processing(result)
                    result['processing_mode'] = 'smart_adaptive'
                else:
                    result['processing_mode'] = 'standard'
                
                # Generate annotated image if requested
                if include_annotation:
                    annotated_base64 = self._generate_annotated_image(image_path, result)
                    if annotated_base64:
                        result['annotated_image_base64'] = annotated_base64
            
            # Clean up temporary file if we created it
            if not temp_file_path and os.path.exists(image_path):
                os.remove(image_path)
            
            if result:
                self.logger.info(f"Image processed - Decision: {result.get('final_decision')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            if not temp_file_path and 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            raise
    
    def _apply_integrated_smart_processing(self, result):
        """Apply integrated smart processing to results - NO SEPARATE FILE NEEDED"""
        try:
            if not self.smart_config['enable_intelligent_filtering']:
                return result
            
            defect_classification = result.get('defect_classification', {})
            
            # Get bounding boxes from result
            if 'defect_analysis' in defect_classification:
                bounding_boxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                defect_statistics = defect_classification['defect_analysis'].get('defect_statistics', {})
            else:
                bounding_boxes = defect_classification.get('bounding_boxes', {})
                defect_statistics = defect_classification.get('defect_statistics', {})
            
            if not bounding_boxes:
                return result
            
            # Apply integrated smart filtering
            filtered_boxes = {}
            filtered_stats = {}
            original_count = sum(len(boxes) for boxes in bounding_boxes.values())
            
            for defect_type, boxes in bounding_boxes.items():
                if not boxes:
                    continue
                
                # Filter by confidence and area
                filtered_type_boxes = self._filter_boxes_smart(boxes, defect_type)
                
                # Apply NMS if enabled
                if self.smart_config['enable_nms']:
                    filtered_type_boxes = self._apply_nms_integrated(filtered_type_boxes)
                
                # Limit detections per type
                max_detections = self.smart_config['max_defects_per_type']
                if len(filtered_type_boxes) > max_detections:
                    filtered_type_boxes = sorted(filtered_type_boxes, 
                                               key=lambda x: x.get('area', 0), reverse=True)[:max_detections]
                
                if filtered_type_boxes:
                    filtered_boxes[defect_type] = filtered_type_boxes
                    filtered_stats[defect_type] = self._recalculate_stats_integrated(filtered_type_boxes, defect_statistics.get(defect_type, {}))
            
            # Update result with filtered data
            if 'defect_analysis' in defect_classification:
                defect_classification['defect_analysis']['bounding_boxes'] = filtered_boxes
                defect_classification['defect_analysis']['defect_statistics'] = filtered_stats
            else:
                defect_classification['bounding_boxes'] = filtered_boxes
                defect_classification['defect_statistics'] = filtered_stats
            
            # Update detected defect types
            result['detected_defect_types'] = list(filtered_boxes.keys())
            
            # Apply smart final decision
            result = self._smart_final_decision_integrated(result)
            
            # Add smart processing info
            filtered_count = sum(len(boxes) for boxes in filtered_boxes.values())
            result['smart_processing'] = {
                'original_detections': original_count,
                'filtered_detections': filtered_count,
                'filtering_applied': True,
                'sensitivity_level': self.smart_config['anomaly_sensitivity'],
                'adaptive_threshold_used': self.smart_config['current_anomaly_threshold']
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integrated smart processing: {e}")
            return result
    
    def _filter_boxes_smart(self, boxes, defect_type):
        """Filter bounding boxes using integrated smart criteria"""
        filtered_boxes = []
        
        min_area_threshold = self.smart_config['min_defect_area_threshold']
        confidence_threshold = self.smart_config['current_defect_threshold']
        
        for bbox in boxes:
            # Check area percentage
            area_percentage = bbox.get('area_percentage', 0)
            if area_percentage < min_area_threshold:
                continue
            
            # Check confidence
            confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
            
            # Adjust threshold for critical defects
            adjusted_threshold = confidence_threshold
            if defect_type in ['missing_component', 'damaged']:
                adjusted_threshold *= 0.9  # More lenient for critical defects
            
            if confidence >= adjusted_threshold:
                # Apply confidence boost for critical defects
                if (self.smart_config['enable_confidence_boosting'] and 
                    defect_type in ['missing_component', 'damaged'] and area_percentage > 5.0):
                    bbox['confidence_boosted'] = True
                    bbox['original_confidence'] = confidence
                    confidence *= self.smart_config['confidence_boost_factor']
                    bbox['confidence'] = min(confidence, 1.0)
                
                filtered_boxes.append(bbox)
        
        return filtered_boxes
    
    def _apply_nms_integrated(self, boxes):
        """Apply integrated Non-Maximum Suppression"""
        if len(boxes) <= 1:
            return boxes
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x.get('confidence', x.get('confidence_score', 0)), reverse=True)
        
        keep = []
        iou_threshold = self.smart_config['nms_iou_threshold']
        
        for box1 in boxes:
            suppress = False
            
            for box2 in keep:
                iou = self._calculate_iou_integrated(box1, box2)
                if iou > iou_threshold:
                    suppress = True
                    break
            
            if not suppress:
                keep.append(box1)
        
        return keep
    
    def _calculate_iou_integrated(self, box1, box2):
        """Calculate IoU for integrated NMS"""
        try:
            # Extract coordinates
            x1_1, y1_1 = box1.get('x', 0), box1.get('y', 0)
            x2_1, y2_1 = x1_1 + box1.get('width', 0), y1_1 + box1.get('height', 0)
            
            x1_2, y1_2 = box2.get('x', 0), box2.get('y', 0)
            x2_2, y2_2 = x1_2 + box2.get('width', 0), y1_2 + box2.get('height', 0)
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _recalculate_stats_integrated(self, boxes, original_stats):
        """Recalculate statistics for filtered boxes"""
        if not boxes:
            return original_stats
        
        confidences = [box.get('confidence', box.get('confidence_score', 0)) for box in boxes]
        areas = [box.get('area', 0) for box in boxes]
        
        new_stats = original_stats.copy()
        new_stats.update({
            'num_regions': len(boxes),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'total_area': sum(areas),
            'avg_area': np.mean(areas) if areas else 0,
            'smart_filtered': True
        })
        
        return new_stats
    
    def _smart_final_decision_integrated(self, result):
        """Make integrated smart final decision"""
        try:
            # Get anomaly results
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            
            # Use adaptive threshold
            adaptive_threshold = self.smart_config['current_anomaly_threshold']
            
            # Get defect information
            detected_defects = result.get('detected_defect_types', [])
            
            # Check for critical defects
            has_critical_defects = False
            if result.get('defect_classification'):
                defect_class = result['defect_classification']
                if 'defect_analysis' in defect_class:
                    bboxes = defect_class['defect_analysis'].get('bounding_boxes', {})
                else:
                    bboxes = defect_class.get('bounding_boxes', {})
                
                for defect_type, boxes in bboxes.items():
                    for box in boxes:
                        area_pct = box.get('area_percentage', 0)
                        if defect_type in ['missing_component', 'damaged'] and area_pct > 5.0:
                            has_critical_defects = True
                            break
                        elif area_pct > 10.0:
                            has_critical_defects = True
                            break
            
            # Smart decision logic
            is_anomalous_adaptive = anomaly_score > adaptive_threshold
            has_significant_defects = len(detected_defects) > 0
            
            if has_critical_defects:
                final_decision = 'DEFECT'
                decision_reason = 'critical_defects_detected'
            elif is_anomalous_adaptive and has_significant_defects:
                final_decision = 'DEFECT'
                decision_reason = 'anomaly_and_defects'
            elif is_anomalous_adaptive and anomaly_score > (adaptive_threshold + 0.2):
                final_decision = 'DEFECT'
                decision_reason = 'high_anomaly_score'
            elif has_significant_defects and len(detected_defects) >= 2:
                final_decision = 'DEFECT'
                decision_reason = 'multiple_defects'
            else:
                final_decision = 'GOOD'
                decision_reason = 'no_significant_issues'
            
            # Update result
            result['final_decision'] = final_decision
            result['smart_decision'] = {
                'reason': decision_reason,
                'adaptive_threshold_used': adaptive_threshold,
                'original_threshold': 0.3,
                'anomaly_score': anomaly_score,
                'is_anomalous_adaptive': is_anomalous_adaptive,
                'detected_defects_count': len(detected_defects),
                'has_critical_defects': has_critical_defects,
                'confidence_level': 'high' if abs(anomaly_score - adaptive_threshold) > 0.2 else 'medium'
            }
            
            # Update anomaly detection with adaptive threshold
            anomaly_detection['threshold_used'] = adaptive_threshold
            anomaly_detection['decision'] = final_decision
            result['anomaly_detection'] = anomaly_detection
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integrated smart final decision: {e}")
            return result
    
    def process_frame(self, image_data, filename, temp_file_path, fast_mode=True, include_annotation=True, use_smart_processing=False):
        """Process frame with optional integrated smart processing"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing frame (smart: {use_smart_processing}, fast: {fast_mode}): {filename}")
            
            # For frames, we can use the same logic as single image but with optimizations
            if use_smart_processing and self.smart_config['smart_enabled']:
                # Use smart processing
                result = self.process_single_image(image_data, filename, temp_file_path, include_annotation, use_smart_processing=True)
                result['frame_mode'] = True
            else:
                # Use standard frame processing (existing logic)
                if fast_mode:
                    result = self.detector.detect_anomaly(temp_file_path)
                    if result:
                        result['processing_mode'] = 'standard'
                        result['frame_mode'] = True
                        result['fast_mode'] = True
                else:
                    result = self.detector.process_image(temp_file_path)
                    if result:
                        result['processing_mode'] = 'standard'
                        result['frame_mode'] = True
                        result['fast_mode'] = False
                
                # Generate annotated image if requested
                if include_annotation and result:
                    annotated_base64 = self._generate_annotated_image(temp_file_path, result)
                    if annotated_base64:
                        result['annotated_image_base64'] = annotated_base64
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            raise
    
    def _generate_annotated_image(self, image_path, result):
        """Generate annotated image (existing method - unchanged)"""
        try:
            from utils.stateless_visualization import create_annotated_image_base64
            return create_annotated_image_base64(image_path, result)
            
        except ImportError:
            import cv2
            import base64
            
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            annotated_image = self._annotate_image_with_openai_insights(image, result, False)
            
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating annotated image: {e}")
            return None
    
    def _annotate_image_with_openai_insights(self, image, result, fast_mode=False):
        """Add annotations (existing method - unchanged)"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            # Get decision and score
            decision = result.get('final_decision', 'UNKNOWN')
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            
            # Add border based on decision
            if decision == 'GOOD':
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated, "GOOD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif decision == 'DEFECT':
                annotated = cv2.copyMakeBorder(annotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated, "DEFECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Draw defect bounding boxes if available
                if result.get('defect_classification'):
                    self._draw_bounding_boxes(annotated, result['defect_classification'])
            
            # Add processing info
            cv2.putText(annotated, f"Score: {anomaly_score:.3f}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add smart processing info if available
            processing_mode = result.get('processing_mode', 'standard')
            cv2.putText(annotated, f"Mode: {processing_mode}", (20, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error annotating image: {e}")
            return image
    
    def _draw_bounding_boxes(self, image, defect_classification):
        """Draw bounding boxes (existing method - unchanged)"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            defect_analysis = defect_classification.get('defect_analysis', {})
            bounding_boxes = defect_analysis.get('bounding_boxes', {})
            
            if not bounding_boxes:
                bounding_boxes = defect_classification.get('bounding_boxes', {})
            
            for defect_type, boxes in bounding_boxes.items():
                defect_class_id = None
                for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
                    if class_name == defect_type:
                        defect_class_id = class_id
                        break
                
                if defect_class_id is not None and defect_class_id in DEFECT_COLORS:
                    color = DEFECT_COLORS[defect_class_id]
                else:
                    default_colors = [(255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
                    color = default_colors[hash(defect_type) % len(default_colors)]
                
                for bbox in boxes:
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    
                    # Draw rectangle with enhanced info for smart mode
                    thickness = 3 if bbox.get('confidence_boosted') else 2
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                    
                    label = defect_type.upper()
                    if bbox.get('confidence_boosted'):
                        label += "*"  # Mark boosted confidence
                    
                    cv2.putText(image, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            self.logger.error(f"Error drawing bounding boxes: {e}")
    
    def update_thresholds(self, new_thresholds):
        """Update detection thresholds (existing method - enhanced)"""
        try:
            for key, value in new_thresholds.items():
                if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        raise ValueError(f"Invalid value for {key}: must be between 0 and 1")
            
            self.config.update(new_thresholds)
            self.logger.info(f"Thresholds updated: {new_thresholds}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return False
    
    def get_thresholds(self):
        """Get current detection thresholds including smart settings"""
        return {
            'standard_thresholds': self.config,
            'smart_settings': self.get_smart_config(),
            'configurable': True,
            'storage': 'in-memory',
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_current_load(self):
        """Get current system load (existing method)"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'status': 'normal'
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'status': 'monitoring_unavailable'
            }
    
    def _get_memory_usage(self):
        """Get memory usage information (existing method)"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent_used': memory.percent,
                'mode': 'integrated_smart'
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'mode': 'integrated_smart'
            }
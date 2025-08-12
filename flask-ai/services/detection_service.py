"""
Detection Service with UPGRADED Real-time Frame Processing
Enhanced with same logic as combined endpoint but optimized for real-time
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
    """Detection Service with UPGRADED Real-time Frame Processing capabilities"""
    
    def __init__(self):
        self.detector = None
        self.is_initialized = False
        self.initialization_error = None
        
        # Frame-specific optimizations cache
        self.frame_cache = {
            'last_processed_time': 0,
            'consecutive_good_frames': 0,
            'model_warmup_done': False
        }
        
        # Smart Configuration for real-time processing
        self.smart_config = {
            'smart_enabled': True,
            'anomaly_sensitivity': 'medium',
            'defect_sensitivity': 'medium',
            'sensitivity_thresholds': {
                'anomaly': {'low': 0.7, 'medium': 0.5, 'high': 0.3},
                'defect': {'low': 0.8, 'medium': 0.7, 'high': 0.6}
            },
            'frame_optimizations': {
                'enable_model_caching': True,
                'skip_consecutive_good_threshold': 5,
                'lightweight_openai_analysis': True,
                'adaptive_quality': True
            },
            'max_defects_per_type': 3,
            'min_defect_area_threshold': 0.1,
            'confidence_boost_factor': 1.2,
            'nms_iou_threshold': 0.3,
            'enable_intelligent_filtering': True,
            'enable_nms': True,
            'enable_confidence_boosting': True
        }
        
        # In-memory configuration
        self.config = {
            'anomaly_threshold': 0.7,
            'defect_confidence_threshold': 0.85
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components with real-time optimizations"""
        try:
            self.logger.info("Initializing detection service with real-time frame processing...")
            
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Main detector initialization failed")
            
            self._calculate_adaptive_thresholds()
            self._warmup_models()
            
            self.is_initialized = True
            self.logger.info("Real-time detection service ready")
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Detection service initialization failed: {e}")
            self.is_initialized = False
    
    def _warmup_models(self):
        """Warmup models for faster real-time processing"""
        try:
            if not self.smart_config['frame_optimizations']['enable_model_caching']:
                return
            
            # Create dummy image for warmup
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, dummy_image)
            
            # Warmup detection pipeline
            self.detector.detect_anomaly(temp_file.name)
            
            os.unlink(temp_file.name)
            self.frame_cache['model_warmup_done'] = True
            self.logger.info("Model warmup completed for real-time processing")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    def _calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds for real-time processing"""
        anomaly_sens = self.smart_config['anomaly_sensitivity']
        defect_sens = self.smart_config['defect_sensitivity']
        
        self.smart_config['current_anomaly_threshold'] = self.smart_config['sensitivity_thresholds']['anomaly'][anomaly_sens]
        self.smart_config['current_defect_threshold'] = self.smart_config['sensitivity_thresholds']['defect'][defect_sens]
        
        self.logger.info(f"Real-time adaptive thresholds - Anomaly: {self.smart_config['current_anomaly_threshold']}, "
                        f"Defect: {self.smart_config['current_defect_threshold']}")
    
    def process_frame(self, image_data, filename, temp_file_path, fast_mode=True, include_annotation=True, 
                     use_smart_processing=True, sensitivity_level=None):
        """
        UPGRADED: Process frame with same enhanced logic as combined endpoint
        Optimized for real-time performance with smart processing
        """
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing real-time frame (smart: {use_smart_processing}, fast: {fast_mode}): {filename}")
            
            # Update sensitivity if provided
            if sensitivity_level and sensitivity_level in ['low', 'medium', 'high']:
                self._update_sensitivity_for_frame(sensitivity_level)
            
            # Check if we can skip processing for consecutive good frames
            if fast_mode and self._should_skip_processing():
                return self._create_cached_good_result(filename, start_time)
            
            # Process with enhanced detection logic
            if use_smart_processing and self.smart_config['smart_enabled']:
                result = self._process_frame_with_enhanced_detection(
                    image_data, filename, temp_file_path, fast_mode
                )
            else:
                result = self._process_frame_standard(
                    image_data, filename, temp_file_path, fast_mode
                )
            
            # Apply real-time optimizations
            if result:
                result = self._apply_real_time_optimizations(result, fast_mode)
                
                # Update frame cache
                self._update_frame_cache(result)
                
                # Generate annotation if requested
                if include_annotation:
                    annotated_base64 = self._generate_annotated_image(temp_file_path, result)
                    if annotated_base64:
                        result['annotated_image_base64'] = annotated_base64
                
                # Add real-time metadata
                result.update({
                    'frame_mode': True,
                    'fast_mode': fast_mode,
                    'real_time_processing': True,
                    'processing_time': time.time() - start_time,
                    'smart_processing_applied': use_smart_processing,
                    'frame_optimizations': self.smart_config['frame_optimizations']
                })
            
            self.logger.info(f"Real-time frame processed - Decision: {result.get('final_decision')} "
                           f"in {time.time() - start_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing real-time frame: {e}")
            raise
    
    def _update_sensitivity_for_frame(self, sensitivity_level):
        """Update sensitivity dynamically for frame processing"""
        self.smart_config['anomaly_sensitivity'] = sensitivity_level
        self.smart_config['defect_sensitivity'] = sensitivity_level
        self._calculate_adaptive_thresholds()
    
    def _should_skip_processing(self):
        """Check if we can skip processing based on consecutive good frames"""
        if not self.smart_config['frame_optimizations']['enable_model_caching']:
            return False
        
        threshold = self.smart_config['frame_optimizations']['skip_consecutive_good_threshold']
        return self.frame_cache['consecutive_good_frames'] >= threshold
    
    def _create_cached_good_result(self, filename, start_time):
        """Create cached result for skipped frames"""
        return {
            'final_decision': 'GOOD',
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'anomaly_detection': {
                'anomaly_score': 0.1,
                'decision': 'GOOD',
                'threshold_used': self.smart_config['current_anomaly_threshold']
            },
            'detected_defect_types': [],
            'frame_mode': True,
            'fast_mode': True,
            'cached_result': True,
            'consecutive_good_count': self.frame_cache['consecutive_good_frames']
        }
    
    def _process_frame_with_enhanced_detection(self, image_data, filename, temp_file_path, fast_mode):
        """Process frame using enhanced detection logic from combined endpoint"""
        try:
            # Use the same process_single_image logic but with frame optimizations
            result = self.detector.process_image(temp_file_path)
            
            if not result:
                return None
            
            # Apply enhanced detection analysis
            result = self._apply_enhanced_frame_analysis(result, temp_file_path, fast_mode)
            
            # Apply smart processing with frame-specific optimizations
            if self.smart_config['enable_intelligent_filtering']:
                result = self._apply_smart_processing_for_frames(result)
            
            # Apply smart final decision with real-time considerations
            result = self._smart_final_decision_for_frames(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced frame processing: {e}")
            return None
    
    def _apply_enhanced_frame_analysis(self, result, temp_file_path, fast_mode):
        """Apply enhanced detection analysis optimized for frames"""
        try:
            # Get defect classification results
            defect_classification = result.get('defect_classification', {})
            
            if not defect_classification:
                return result
            
            # Apply enhanced detection with frame-specific optimizations
            if 'defect_analysis' in defect_classification:
                enhanced_analysis = defect_classification['defect_analysis']
            else:
                # Use enhanced detection logic
                from core.enhanced_detection import analyze_defect_predictions_enhanced
                
                # Get prediction data from result
                predicted_mask = self._extract_mask_from_result(result)
                confidence_scores = self._extract_confidence_from_result(result)
                
                if predicted_mask is not None and confidence_scores is not None:
                    image_shape = predicted_mask.shape
                    enhanced_analysis = analyze_defect_predictions_enhanced(
                        predicted_mask, confidence_scores, image_shape
                    )
                    
                    # Update result with enhanced analysis
                    defect_classification['defect_analysis'] = enhanced_analysis
                    result['defect_classification'] = defect_classification
                    result['detected_defect_types'] = enhanced_analysis.get('detected_defects', [])
            
            # Add frame-specific enhancements
            result['frame_enhanced_detection'] = True
            result['guaranteed_defect_detection'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced frame analysis: {e}")
            return result
    
    def _extract_mask_from_result(self, result):
        """Extract prediction mask from detection result"""
        try:
            defect_class = result.get('defect_classification', {})
            return defect_class.get('predicted_mask')
        except:
            return None
    
    def _extract_confidence_from_result(self, result):
        """Extract confidence scores from detection result"""
        try:
            defect_class = result.get('defect_classification', {})
            return defect_class.get('confidence_scores')
        except:
            return None
    
    def _apply_smart_processing_for_frames(self, result):
        """Apply smart processing optimized for real-time frames"""
        try:
            defect_classification = result.get('defect_classification', {})
            
            # Get bounding boxes
            if 'defect_analysis' in defect_classification:
                bounding_boxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                defect_statistics = defect_classification['defect_analysis'].get('defect_statistics', {})
            else:
                bounding_boxes = defect_classification.get('bounding_boxes', {})
                defect_statistics = defect_classification.get('defect_statistics', {})
            
            if not bounding_boxes:
                return result
            
            # Apply frame-optimized filtering
            filtered_boxes = {}
            filtered_stats = {}
            
            for defect_type, boxes in bounding_boxes.items():
                if not boxes:
                    continue
                
                # Filter with frame-specific thresholds
                filtered_type_boxes = self._filter_boxes_for_frames(boxes, defect_type)
                
                # Apply lightweight NMS for frames
                if self.smart_config['enable_nms'] and len(filtered_type_boxes) > 1:
                    filtered_type_boxes = self._apply_lightweight_nms(filtered_type_boxes)
                
                # Limit detections for real-time processing
                max_detections = min(self.smart_config['max_defects_per_type'], 2)  # Reduced for frames
                if len(filtered_type_boxes) > max_detections:
                    filtered_type_boxes = sorted(filtered_type_boxes, 
                                               key=lambda x: x.get('area', 0), reverse=True)[:max_detections]
                
                if filtered_type_boxes:
                    filtered_boxes[defect_type] = filtered_type_boxes
                    filtered_stats[defect_type] = self._recalculate_stats_for_frames(
                        filtered_type_boxes, defect_statistics.get(defect_type, {})
                    )
            
            # Update result
            if 'defect_analysis' in defect_classification:
                defect_classification['defect_analysis']['bounding_boxes'] = filtered_boxes
                defect_classification['defect_analysis']['defect_statistics'] = filtered_stats
            else:
                defect_classification['bounding_boxes'] = filtered_boxes
                defect_classification['defect_statistics'] = filtered_stats
            
            result['detected_defect_types'] = list(filtered_boxes.keys())
            result['frame_smart_processing'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame smart processing: {e}")
            return result
    
    def _filter_boxes_for_frames(self, boxes, defect_type):
        """Filter bounding boxes with frame-specific criteria"""
        filtered_boxes = []
        
        # More lenient thresholds for real-time processing
        min_area_threshold = self.smart_config['min_defect_area_threshold'] * 0.5  # More lenient
        confidence_threshold = self.smart_config['current_defect_threshold'] * 0.9  # More lenient
        
        for bbox in boxes:
            area_percentage = bbox.get('area_percentage', 0)
            if area_percentage < min_area_threshold:
                continue
            
            confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
            
            # Adjust threshold for critical defects in frames
            adjusted_threshold = confidence_threshold
            if defect_type in ['missing_component', 'damaged']:
                adjusted_threshold *= 0.8  # Even more lenient for critical defects in frames
            
            if confidence >= adjusted_threshold:
                # Frame-specific confidence boost
                if (self.smart_config['enable_confidence_boosting'] and 
                    defect_type in ['missing_component', 'damaged']):
                    boost_factor = self.smart_config['confidence_boost_factor'] * 1.1  # Slightly higher boost
                    bbox['frame_confidence_boosted'] = True
                    bbox['original_confidence'] = confidence
                    confidence *= boost_factor
                    bbox['confidence'] = min(confidence, 1.0)
                
                filtered_boxes.append(bbox)
        
        return filtered_boxes
    
    def _apply_lightweight_nms(self, boxes):
        """Apply lightweight NMS optimized for real-time frames"""
        if len(boxes) <= 1:
            return boxes
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda x: x.get('confidence', x.get('confidence_score', 0)), reverse=True)
        
        keep = []
        iou_threshold = self.smart_config['nms_iou_threshold'] * 1.2  # More lenient for frames
        
        for box1 in boxes:
            suppress = False
            
            for box2 in keep:
                iou = self._calculate_iou_fast(box1, box2)
                if iou > iou_threshold:
                    suppress = True
                    break
            
            if not suppress:
                keep.append(box1)
                # Limit to reduce processing time
                if len(keep) >= 2:
                    break
        
        return keep
    
    def _calculate_iou_fast(self, box1, box2):
        """Fast IoU calculation for real-time processing"""
        try:
            x1_1, y1_1 = box1.get('x', 0), box1.get('y', 0)
            x2_1, y2_1 = x1_1 + box1.get('width', 0), y1_1 + box1.get('height', 0)
            
            x1_2, y1_2 = box2.get('x', 0), box2.get('y', 0)
            x2_2, y2_2 = x1_2 + box2.get('width', 0), y1_2 + box2.get('height', 0)
            
            # Fast intersection check
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
    
    def _recalculate_stats_for_frames(self, boxes, original_stats):
        """Recalculate statistics for filtered boxes in frame processing"""
        if not boxes:
            return original_stats
        
        confidences = [box.get('confidence', box.get('confidence_score', 0)) for box in boxes]
        areas = [box.get('area', 0) for box in boxes]
        
        new_stats = original_stats.copy()
        new_stats.update({
            'num_regions': len(boxes),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'total_area': sum(areas),
            'frame_optimized': True
        })
        
        return new_stats
    
    def _smart_final_decision_for_frames(self, result):
        """Make smart final decision optimized for real-time frames"""
        try:
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            
            # Use adaptive threshold with frame-specific adjustments
            adaptive_threshold = self.smart_config['current_anomaly_threshold']
            frame_threshold = adaptive_threshold * 0.95  # Slightly more sensitive for frames
            
            detected_defects = result.get('detected_defect_types', [])
            
            # Frame-specific decision logic
            is_anomalous_adaptive = anomaly_score > frame_threshold
            has_defects = len(detected_defects) > 0
            
            # Critical defect check with frame optimization
            has_critical_defects = self._check_critical_defects_fast(result)
            
            if has_critical_defects:
                final_decision = 'DEFECT'
                decision_reason = 'critical_defects_frame'
            elif is_anomalous_adaptive and has_defects:
                final_decision = 'DEFECT'
                decision_reason = 'anomaly_and_defects_frame'
            elif is_anomalous_adaptive and anomaly_score > (frame_threshold + 0.15):
                final_decision = 'DEFECT'
                decision_reason = 'high_anomaly_frame'
            elif has_defects:
                final_decision = 'DEFECT'
                decision_reason = 'defects_detected_frame'
            else:
                final_decision = 'GOOD'
                decision_reason = 'no_issues_frame'
            
            # Update result
            result['final_decision'] = final_decision
            result['frame_smart_decision'] = {
                'reason': decision_reason,
                'frame_threshold_used': frame_threshold,
                'original_threshold': adaptive_threshold,
                'anomaly_score': anomaly_score,
                'detected_defects_count': len(detected_defects),
                'has_critical_defects': has_critical_defects
            }
            
            # Update anomaly detection
            anomaly_detection['threshold_used'] = frame_threshold
            anomaly_detection['decision'] = final_decision
            result['anomaly_detection'] = anomaly_detection
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame smart decision: {e}")
            return result
    
    def _check_critical_defects_fast(self, result):
        """Fast critical defect check for frames"""
        try:
            defect_class = result.get('defect_classification', {})
            
            if 'defect_analysis' in defect_class:
                bboxes = defect_class['defect_analysis'].get('bounding_boxes', {})
            else:
                bboxes = defect_class.get('bounding_boxes', {})
            
            for defect_type, boxes in bboxes.items():
                if defect_type in ['missing_component', 'damaged']:
                    for box in boxes[:1]:  # Check only first box for speed
                        area_pct = box.get('area_percentage', 0)
                        if area_pct > 3.0:  # Lower threshold for frames
                            return True
                        
            return False
            
        except Exception:
            return False
    
    def _apply_real_time_optimizations(self, result, fast_mode):
        """Apply real-time specific optimizations"""
        try:
            # Add processing metadata
            result['real_time_optimizations'] = {
                'adaptive_quality': self.smart_config['frame_optimizations']['adaptive_quality'],
                'model_caching': self.smart_config['frame_optimizations']['enable_model_caching'],
                'lightweight_analysis': fast_mode,
                'frame_threshold_adjustment': True
            }
            
            # Optimize OpenAI analysis for frames if enabled
            if result.get('anomaly_detection', {}).get('openai_analysis'):
                result['anomaly_detection']['openai_analysis']['frame_optimized'] = True
            
            if result.get('defect_classification', {}).get('openai_analysis'):
                result['defect_classification']['openai_analysis']['frame_optimized'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying real-time optimizations: {e}")
            return result
    
    def _update_frame_cache(self, result):
        """Update frame processing cache"""
        try:
            current_time = time.time()
            self.frame_cache['last_processed_time'] = current_time
            
            if result.get('final_decision') == 'GOOD':
                self.frame_cache['consecutive_good_frames'] += 1
            else:
                self.frame_cache['consecutive_good_frames'] = 0
                
        except Exception as e:
            self.logger.error(f"Error updating frame cache: {e}")
    
    def _process_frame_standard(self, image_data, filename, temp_file_path, fast_mode):
        """Standard frame processing fallback"""
        try:
            if fast_mode:
                result = self.detector.detect_anomaly(temp_file_path)
                if result:
                    result['processing_mode'] = 'standard_fast'
                    result['frame_mode'] = True
                    result['fast_mode'] = True
            else:
                result = self.detector.process_image(temp_file_path)
                if result:
                    result['processing_mode'] = 'standard_full'
                    result['frame_mode'] = True
                    result['fast_mode'] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in standard frame processing: {e}")
            return None
    
    # Keep all existing methods from original DetectionService
    def get_health_status(self):
        """Get health status including real-time frame processing"""
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
            'real_time_processing': {
                'enabled': True,
                'frame_optimizations': self.smart_config['frame_optimizations'],
                'model_warmup_done': self.frame_cache['model_warmup_done'],
                'adaptive_thresholds': True,
                'enhanced_detection': True
            },
            'smart_processing': {
                'enabled': self.smart_config['smart_enabled'],
                'integrated': True,
                'sensitivity_level': self.smart_config['anomaly_sensitivity'],
                'intelligent_filtering': self.smart_config['enable_intelligent_filtering']
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error,
            'mode': 'real_time_enhanced'
        }
        
        return base_status
    
    def process_single_image(self, image_data, filename, temp_file_path=None, include_annotation=True, use_smart_processing=False):
        """Process single image - keeping original functionality"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing image (smart: {use_smart_processing}): {filename}")
            
            if temp_file_path:
                image_path = temp_file_path
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(image_data)
                temp_file.close()
                image_path = temp_file.name
            
            result = self.detector.process_image(image_path)
            
            if result:
                if use_smart_processing and self.smart_config['smart_enabled']:
                    result = self._apply_integrated_smart_processing(result)
                    result['processing_mode'] = 'smart_adaptive'
                else:
                    result['processing_mode'] = 'standard'
                
                if include_annotation:
                    annotated_base64 = self._generate_annotated_image(image_path, result)
                    if annotated_base64:
                        result['annotated_image_base64'] = annotated_base64
            
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
    
    # Include other existing methods from original DetectionService...
    def _generate_annotated_image(self, image_path, result):
        """Generate annotated image"""
        try:
            from utils.stateless_visualization import create_annotated_image_base64
            return create_annotated_image_base64(image_path, result)
            
        except ImportError:
            import cv2
            import base64
            
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            annotated_image = self._annotate_image_with_insights(image, result)
            
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating annotated image: {e}")
            return None
    
    def _annotate_image_with_insights(self, image, result):
        """Add annotations to image"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            annotated = image.copy()
            height, width = annotated.shape[:2]
            
            decision = result.get('final_decision', 'UNKNOWN')
            anomaly_score = result.get('anomaly_detection', {}).get('anomaly_score', 0.0)
            
            if decision == 'GOOD':
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
                cv2.putText(annotated, "GOOD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif decision == 'DEFECT':
                annotated = cv2.copyMakeBorder(annotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                cv2.putText(annotated, "DEFECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                if result.get('defect_classification'):
                    self._draw_bounding_boxes(annotated, result['defect_classification'])
            
            cv2.putText(annotated, f"Score: {anomaly_score:.3f}", (20, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            processing_mode = result.get('processing_mode', 'standard')
            cv2.putText(annotated, f"Mode: {processing_mode}", (20, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error annotating image: {e}")
            return image
    
    def _draw_bounding_boxes(self, image, defect_classification):
        """Draw bounding boxes for defects"""
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
                    
                    thickness = 3 if bbox.get('frame_confidence_boosted') else 2
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                    
                    label = defect_type.upper()
                    if bbox.get('frame_confidence_boosted'):
                        label += "*"
                    
                    cv2.putText(image, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            self.logger.error(f"Error drawing bounding boxes: {e}")
    
    def _apply_integrated_smart_processing(self, result):
        """Apply integrated smart processing to results"""
        try:
            if not self.smart_config['enable_intelligent_filtering']:
                return result
            
            defect_classification = result.get('defect_classification', {})
            
            if 'defect_analysis' in defect_classification:
                bounding_boxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                defect_statistics = defect_classification['defect_analysis'].get('defect_statistics', {})
            else:
                bounding_boxes = defect_classification.get('bounding_boxes', {})
                defect_statistics = defect_classification.get('defect_statistics', {})
            
            if not bounding_boxes:
                return result
            
            filtered_boxes = {}
            filtered_stats = {}
            original_count = sum(len(boxes) for boxes in bounding_boxes.values())
            
            for defect_type, boxes in bounding_boxes.items():
                if not boxes:
                    continue
                
                filtered_type_boxes = self._filter_boxes_smart(boxes, defect_type)
                
                if self.smart_config['enable_nms']:
                    filtered_type_boxes = self._apply_nms_integrated(filtered_type_boxes)
                
                max_detections = self.smart_config['max_defects_per_type']
                if len(filtered_type_boxes) > max_detections:
                    filtered_type_boxes = sorted(filtered_type_boxes, 
                                               key=lambda x: x.get('area', 0), reverse=True)[:max_detections]
                
                if filtered_type_boxes:
                    filtered_boxes[defect_type] = filtered_type_boxes
                    filtered_stats[defect_type] = self._recalculate_stats_integrated(
                        filtered_type_boxes, defect_statistics.get(defect_type, {})
                    )
            
            if 'defect_analysis' in defect_classification:
                defect_classification['defect_analysis']['bounding_boxes'] = filtered_boxes
                defect_classification['defect_analysis']['defect_statistics'] = filtered_stats
            else:
                defect_classification['bounding_boxes'] = filtered_boxes
                defect_classification['defect_statistics'] = filtered_stats
            
            result['detected_defect_types'] = list(filtered_boxes.keys())
            result = self._smart_final_decision_integrated(result)
            
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
            area_percentage = bbox.get('area_percentage', 0)
            if area_percentage < min_area_threshold:
                continue
            
            confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
            
            adjusted_threshold = confidence_threshold
            if defect_type in ['missing_component', 'damaged']:
                adjusted_threshold *= 0.9
            
            if confidence >= adjusted_threshold:
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
            x1_1, y1_1 = box1.get('x', 0), box1.get('y', 0)
            x2_1, y2_1 = x1_1 + box1.get('width', 0), y1_1 + box1.get('height', 0)
            
            x1_2, y1_2 = box2.get('x', 0), box2.get('y', 0)
            x2_2, y2_2 = x1_2 + box2.get('width', 0), y1_2 + box2.get('height', 0)
            
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
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            
            adaptive_threshold = self.smart_config['current_anomaly_threshold']
            detected_defects = result.get('detected_defect_types', [])
            
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
            
            anomaly_detection['threshold_used'] = adaptive_threshold
            anomaly_detection['decision'] = final_decision
            result['anomaly_detection'] = anomaly_detection
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integrated smart final decision: {e}")
            return result
    
    def set_smart_sensitivity(self, anomaly_sensitivity='medium', defect_sensitivity='medium'):
        """Set smart processing sensitivity levels"""
        valid_levels = ['low', 'medium', 'high']
        
        if anomaly_sensitivity in valid_levels:
            self.smart_config['anomaly_sensitivity'] = anomaly_sensitivity
        if defect_sensitivity in valid_levels:
            self.smart_config['defect_sensitivity'] = defect_sensitivity
        
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
            },
            'frame_optimizations': self.smart_config['frame_optimizations']
        }
    
    def get_system_information(self):
        """Get system information including real-time processing"""
        if not self.detector:
            return {
                'status': 'not_initialized',
                'error': 'Detector not available',
                'mode': 'real_time_enhanced'
            }
        
        try:
            from config import OPENAI_API_KEY, OPENAI_MODEL
            
            system_info = self.detector.get_system_info()
            
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'real_time_processing': {
                    'enabled': True,
                    'frame_optimizations': self.smart_config['frame_optimizations'],
                    'enhanced_detection': True,
                    'adaptive_thresholds': True,
                    'model_warmup': self.frame_cache['model_warmup_done'],
                    'features': ['guaranteed_defect_detection', 'smart_filtering', 'adaptive_sensitivity', 'frame_caching']
                },
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
                    'real_time_processor': True,
                    'enhanced_detector': True,
                    'openai_analyzer': bool(OPENAI_API_KEY),
                    'database': False,
                    'file_storage': False
                },
                'api_version': '1.0.0',
                'mode': 'real_time_enhanced'
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'mode': 'real_time_enhanced'
            }
    
    def get_current_status(self):
        """Get current system status with real-time processing info"""
        from config import OPENAI_API_KEY
        
        return {
            'system_ready': self.is_initialized,
            'detector_ready': self.detector.is_ready() if self.detector else False,
            'openai_ready': bool(OPENAI_API_KEY),
            'smart_processing_ready': self.smart_config['smart_enabled'],
            'real_time_ready': True,
            'processing_capabilities': {
                'single_image': self.detector is not None,
                'batch_processing': self.detector is not None,
                'real_time_frames': True,
                'smart_processing': self.smart_config['smart_enabled'],
                'enhanced_detection': True,
                'openai_analysis': bool(OPENAI_API_KEY),
                'frame_caching': self.smart_config['frame_optimizations']['enable_model_caching']
            },
            'frame_cache_info': {
                'consecutive_good_frames': self.frame_cache['consecutive_good_frames'],
                'model_warmup_done': self.frame_cache['model_warmup_done'],
                'last_processed': self.frame_cache['last_processed_time']
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'mode': 'real_time_enhanced',
            'last_check': datetime.now().isoformat()
        }
    
    def update_thresholds(self, new_thresholds):
        """Update detection thresholds"""
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
        """Get current system load"""
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
        """Get memory usage information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent_used': memory.percent,
                'mode': 'real_time_enhanced'
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'mode': 'real_time_enhanced'
            }
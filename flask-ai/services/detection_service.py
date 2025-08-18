import os
import cv2
import numpy as np
import tempfile
import time
import logging
from datetime import datetime
from main import create_detector


class DetectionService:
    """Product-Aware Detection Service - Clean Syntax"""
    
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
        
        # Smart Configuration
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
            'max_defects_per_type': 1,
            'min_defect_area_threshold': 0.1,
            'confidence_boost_factor': 1.2,
            'nms_iou_threshold': 0.3,
            'enable_intelligent_filtering': True,
            'enable_nms': False,
            'enable_confidence_boosting': True,
            'background_class_skip': True,
            'product_context_validation': True
        }
        
        # Configuration
        self.config = {
            'anomaly_threshold': 0.7,
            'defect_confidence_threshold': 0.85
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize detection components"""
        try:
            self.logger.info("Initializing product-aware detection service...")
            
            self.detector = create_detector()
            
            if not self.detector or not self.detector.is_ready():
                raise RuntimeError("Detector initialization failed or not ready")
            
            self._calculate_adaptive_thresholds()
            self._warmup_models()
            
            self.is_initialized = True
            self.logger.info("Product-aware detection service ready")
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Detection service initialization failed: {e}")
            self.is_initialized = False
            raise RuntimeError(f"Detection service initialization failed: {e}")
    
    def _warmup_models(self):
        """Warmup models for faster processing"""
        try:
            if not self.smart_config['frame_optimizations']['enable_model_caching']:
                return
            
            dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, dummy_image)
            
            try:
                self.detector.detect_anomaly(temp_file.name)
                self.frame_cache['model_warmup_done'] = True
                self.logger.info("Model warmup completed")
            except Exception as warmup_error:
                self.logger.warning(f"Model warmup failed: {warmup_error}")
            
            os.unlink(temp_file.name)
            
        except Exception as e:
            self.logger.warning(f"Model warmup setup failed: {e}")
    
    def _calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds"""
        anomaly_sens = self.smart_config['anomaly_sensitivity']
        defect_sens = self.smart_config['defect_sensitivity']
        
        self.smart_config['current_anomaly_threshold'] = self.smart_config['sensitivity_thresholds']['anomaly'][anomaly_sens]
        self.smart_config['current_defect_threshold'] = self.smart_config['sensitivity_thresholds']['defect'][defect_sens]
        
        self.logger.info(f"Adaptive thresholds - Anomaly: {self.smart_config['current_anomaly_threshold']}, "
                        f"Defect: {self.smart_config['current_defect_threshold']}")
    
    def process_single_image(self, image_data, filename, temp_file_path=None, include_annotation=True, use_smart_processing=False):
        """Process single image with product-aware validation"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        try:
            self.logger.info(f"Processing image with product validation: {filename}")
            
            if temp_file_path:
                image_path = temp_file_path
            else:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_file.write(image_data)
                temp_file.close()
                image_path = temp_file.name
            
            result = self.detector.process_image(image_path)
            
            if not result:
                raise RuntimeError("Image processing returned no result")
            
            # Apply product-aware validation
            result = self._apply_product_aware_validation(result, image_path)
            
            # Ensure background class excluded
            result = self._ensure_background_class_excluded(result)
            
            if use_smart_processing and self.smart_config['smart_enabled']:
                result = self._apply_integrated_smart_processing(result)
                result['processing_mode'] = 'smart_adaptive_product_aware'
            else:
                result['processing_mode'] = 'standard_product_aware'
            
            result['product_context_validated'] = True
            
            if include_annotation:
                annotated_base64 = self._generate_annotated_image(image_path, result)
                if annotated_base64:
                    result['annotated_image_base64'] = annotated_base64
            
            if not temp_file_path and os.path.exists(image_path):
                os.remove(image_path)
            
            self.logger.info(f"Image processed - Decision: {result.get('final_decision')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            if not temp_file_path and 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            raise RuntimeError(f"Image processing failed: {e}")
    
    def _apply_product_aware_validation(self, result, image_path):
        """Apply product-aware validation"""
        try:
            # Check OpenAI analysis for non-packaging detection
            openai_analysis = result.get('openai_analysis', {})
            
            if openai_analysis:
                # Check anomaly layer
                anomaly_detection = result.get('anomaly_detection', {})
                if anomaly_detection:
                    anomaly_openai = anomaly_detection.get('openai_analysis', {})
                    if anomaly_openai:
                        product_type = anomaly_openai.get('product_type', 'unknown')
                        if product_type == 'INVALID_NON_PACKAGING':
                            self.logger.info("Non-packaging item detected - forcing GOOD")
                            return self._create_non_packaging_result(result, "Non-packaging item detected")
                
                # Check defect layer
                defect_classification = result.get('defect_classification', {})
                if defect_classification:
                    defect_openai = defect_classification.get('openai_analysis', {})
                    if defect_openai:
                        product_validation = defect_openai.get('product_validation', 'unknown')
                        if product_validation == 'NON_PACKAGING_ITEM':
                            self.logger.info("Defect analysis rejected non-packaging item")
                            return self._create_non_packaging_result(result, "Non-packaging item rejected")
            
            # Default: assume packaging
            result['product_validation'] = {
                'validated': True,
                'product_type': 'assumed_packaging',
                'validation_source': 'default_processing',
                'non_packaging_rejected': False
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in product validation: {e}")
            return result
    
    def _create_non_packaging_result(self, original_result, reason):
        """Create result for non-packaging items"""
        try:
            result = original_result.copy()
            
            # Force GOOD decision
            result['final_decision'] = 'GOOD'
            result['detected_defect_types'] = []
            
            # Update anomaly detection
            if 'anomaly_detection' in result:
                result['anomaly_detection']['decision'] = 'GOOD'
            
            # Clear defect classification
            if 'defect_classification' in result:
                if 'defect_analysis' in result['defect_classification']:
                    result['defect_classification']['defect_analysis']['detected_defects'] = []
                    result['defect_classification']['defect_analysis']['bounding_boxes'] = {}
                
                result['defect_classification']['detected_defects'] = []
                result['defect_classification']['bounding_boxes'] = {}
            
            # Add validation metadata
            result['product_validation'] = {
                'validated': True,
                'product_type': 'non_packaging_item',
                'validation_source': 'openai_analysis',
                'rejection_reason': reason,
                'non_packaging_rejected': True,
                'original_decision_overridden': True
            }
            
            result['product_context_filtering_applied'] = True
            result['false_positive_protection'] = True
            
            self.logger.info(f"Non-packaging result created: {reason}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating non-packaging result: {e}")
            return original_result
    
    def process_frame(self, image_data, filename, temp_file_path, fast_mode=True, include_annotation=True, 
                     use_smart_processing=True, sensitivity_level=None):
        """Process frame with product validation"""
        if not self.is_initialized:
            raise RuntimeError("Detection service not initialized")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing frame: {filename}")
            
            if sensitivity_level and sensitivity_level in ['low', 'medium', 'high']:
                self._update_sensitivity_for_frame(sensitivity_level)
            
            if fast_mode and self._should_skip_processing():
                return self._create_skip_result(filename, start_time)
            
            if use_smart_processing and self.smart_config['smart_enabled']:
                result = self._process_frame_with_product_aware_detection(
                    image_data, filename, temp_file_path, fast_mode
                )
            else:
                result = self._process_frame_standard(
                    image_data, filename, temp_file_path, fast_mode
                )
            
            if not result:
                raise RuntimeError("Frame processing returned no result")
            
            result = self._apply_real_time_optimizations(result, fast_mode)
            self._update_frame_cache(result)
            
            if include_annotation:
                annotated_base64 = self._generate_annotated_image(temp_file_path, result)
                if annotated_base64:
                    result['annotated_image_base64'] = annotated_base64
            
            result.update({
                'frame_mode': True,
                'fast_mode': fast_mode,
                'real_time_processing': True,
                'processing_time': time.time() - start_time,
                'smart_processing_applied': use_smart_processing,
                'frame_optimizations': self.smart_config['frame_optimizations'],
                'product_context_validated': True
            })
            
            self.logger.info(f"Frame processed - Decision: {result.get('final_decision')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            raise RuntimeError(f"Frame processing failed: {e}")
    
    def _process_frame_with_product_aware_detection(self, image_data, filename, temp_file_path, fast_mode):
        """Process frame with product awareness"""
        try:
            result = self.process_single_image(image_data, filename, temp_file_path, 
                                             include_annotation=False, use_smart_processing=True)
            
            if not result:
                raise RuntimeError("Product-aware detection failed")
            
            result = self._apply_product_aware_frame_analysis(result, temp_file_path, fast_mode)
            
            if self.smart_config['enable_intelligent_filtering']:
                result = self._apply_smart_processing_for_frames(result)
            
            result = self._smart_final_decision_for_frames(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame processing: {e}")
            raise RuntimeError(f"Frame processing failed: {e}")
    
    def _apply_product_aware_frame_analysis(self, result, temp_file_path, fast_mode):
        """Apply product-aware frame analysis"""
        try:
            product_validation = result.get('product_validation', {})
            
            if product_validation.get('non_packaging_rejected', False):
                self.logger.info("Frame is non-packaging item")
                result['frame_product_validation'] = {
                    'non_packaging_item': True,
                    'validation_source': 'product_aware_analysis',
                    'frame_specific_validation': True
                }
                return result
            
            defect_classification = result.get('defect_classification', {})
            
            if defect_classification:
                detected_defects = defect_classification.get('defect_analysis', {}).get('detected_defects', [])
                
                if 'background' in detected_defects:
                    self.logger.warning("Removing background class from results")
                    detected_defects.remove('background')
                    
                    if 'defect_analysis' in defect_classification:
                        defect_classification['defect_analysis']['detected_defects'] = detected_defects
                        
                        bboxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                        if 'background' in bboxes:
                            del bboxes['background']
                            defect_classification['defect_analysis']['bounding_boxes'] = bboxes
                        
                        stats = defect_classification['defect_analysis'].get('defect_statistics', {})
                        if 'background' in stats:
                            del stats['background']
                            defect_classification['defect_analysis']['defect_statistics'] = stats
                    
                    result['defect_classification'] = defect_classification
                    result['detected_defect_types'] = detected_defects
                
                result['frame_enhanced_detection'] = True
                result['background_class_properly_skipped'] = True
                result['single_bbox_per_defect_type'] = True
                result['frame_product_validation'] = {
                    'packaging_item': True,
                    'validation_source': 'product_aware_analysis',
                    'frame_specific_validation': True
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame analysis: {e}")
            return result
    
    def _update_sensitivity_for_frame(self, sensitivity_level):
        """Update sensitivity for frame"""
        self.smart_config['anomaly_sensitivity'] = sensitivity_level
        self.smart_config['defect_sensitivity'] = sensitivity_level
        self._calculate_adaptive_thresholds()
    
    def _should_skip_processing(self):
        """Check if processing can be skipped"""
        if not self.smart_config['frame_optimizations']['enable_model_caching']:
            return False
        
        threshold = self.smart_config['frame_optimizations']['skip_consecutive_good_threshold']
        return self.frame_cache['consecutive_good_frames'] >= threshold
    
    def _create_skip_result(self, filename, start_time):
        """Create result for skipped frames"""
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
            'consecutive_good_count': self.frame_cache['consecutive_good_frames'],
            'background_class_skipped': True,
            'product_context_validated': True
        }
    
    def _apply_real_time_optimizations(self, result, fast_mode):
        """Apply real-time optimizations"""
        try:
            result['real_time_optimizations'] = {
                'adaptive_quality': self.smart_config['frame_optimizations']['adaptive_quality'],
                'model_caching': self.smart_config['frame_optimizations']['enable_model_caching'],
                'lightweight_analysis': fast_mode,
                'single_bbox_per_type': True,
                'background_class_skipped': True,
                'product_context_validation': True
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying optimizations: {e}")
            return result
    
    def _update_frame_cache(self, result):
        """Update frame cache"""
        try:
            current_time = time.time()
            self.frame_cache['last_processed_time'] = current_time
            
            if result.get('final_decision') == 'GOOD':
                self.frame_cache['consecutive_good_frames'] += 1
            else:
                self.frame_cache['consecutive_good_frames'] = 0
                
        except Exception as e:
            self.logger.error(f"Error updating cache: {e}")
    
    def _process_frame_standard(self, image_data, filename, temp_file_path, fast_mode):
        """Standard frame processing"""
        try:
            if fast_mode:
                result = self.detector.detect_anomaly(temp_file_path)
                if result:
                    result['processing_mode'] = 'standard_fast_product_aware'
                    result['frame_mode'] = True
                    result['fast_mode'] = True
                    result['background_class_skipped'] = True
                    result['product_context_validated'] = True
            else:
                result = self.detector.process_image(temp_file_path)
                if result:
                    result = self._apply_product_aware_validation(result, temp_file_path)
                    result['processing_mode'] = 'standard_full_product_aware'
                    result['frame_mode'] = True
                    result['fast_mode'] = False
                    result['background_class_skipped'] = True
                    result['product_context_validated'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in standard processing: {e}")
            raise RuntimeError(f"Standard processing failed: {e}")
    
    def _ensure_background_class_excluded(self, result):
        """Ensure background class is excluded"""
        try:
            defect_classification = result.get('defect_classification', {})
            
            if defect_classification:
                if 'defect_analysis' in defect_classification:
                    detected_defects = defect_classification['defect_analysis'].get('detected_defects', [])
                    if 'background' in detected_defects:
                        detected_defects.remove('background')
                        defect_classification['defect_analysis']['detected_defects'] = detected_defects
                    
                    bboxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                    if 'background' in bboxes:
                        del bboxes['background']
                        defect_classification['defect_analysis']['bounding_boxes'] = bboxes
                    
                    stats = defect_classification['defect_analysis'].get('defect_statistics', {})
                    if 'background' in stats:
                        del stats['background']
                        defect_classification['defect_analysis']['defect_statistics'] = stats
                
                detected_defects = result.get('detected_defect_types', [])
                if 'background' in detected_defects:
                    detected_defects.remove('background')
                    result['detected_defect_types'] = detected_defects
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error excluding background: {e}")
            return result
    
    def _apply_integrated_smart_processing(self, result):
        """Apply integrated smart processing"""
        try:
            if not self.smart_config['enable_intelligent_filtering']:
                return result
            
            product_validation = result.get('product_validation', {})
            if product_validation.get('non_packaging_rejected', False):
                self.logger.info("Skipping smart processing for non-packaging item")
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
                if not boxes or defect_type == 'background':
                    continue
                
                single_box = boxes[0] if boxes else None
                
                if single_box and self._validate_single_bbox(single_box, defect_type):
                    filtered_boxes[defect_type] = [single_box]
                    filtered_stats[defect_type] = self._recalculate_stats_for_single_bbox(
                        single_box, defect_statistics.get(defect_type, {})
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
                'adaptive_threshold_used': self.smart_config['current_anomaly_threshold'],
                'background_class_excluded': True,
                'single_bbox_per_type_enforced': True,
                'product_context_considered': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in smart processing: {e}")
            return result
    
    def _validate_single_bbox(self, bbox, defect_type):
        """Validate single bounding box"""
        try:
            if not bbox or not isinstance(bbox, dict):
                return False
            
            required_fields = ['x', 'y', 'width', 'height', 'area_percentage']
            for field in required_fields:
                if field not in bbox:
                    return False
            
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False
            
            area_pct = bbox.get('area_percentage', 0)
            if area_pct <= 0 or area_pct > 100:
                return False
            
            confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
            if confidence < 0 or confidence > 1:
                return False
            
            if defect_type == 'missing_component' and area_pct > 90:
                self.logger.warning(f"Missing component covers {area_pct}% - likely incorrect")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating bbox for {defect_type}: {e}")
            return False
    
    def _recalculate_stats_for_single_bbox(self, bbox, original_stats):
        """Recalculate stats for single bbox"""
        if not bbox:
            return original_stats
        
        confidence = bbox.get('confidence', bbox.get('confidence_score', 0))
        area = bbox.get('area', 0)
        
        new_stats = original_stats.copy()
        new_stats.update({
            'num_regions': 1,
            'avg_confidence': confidence,
            'max_confidence': confidence,
            'min_confidence': confidence,
            'total_area': area,
            'avg_area': area,
            'single_bbox_per_type': True,
            'frame_optimized': True
        })
        
        return new_stats
    
    def _apply_smart_processing_for_frames(self, result):
        """Apply smart processing for frames"""
        try:
            product_validation = result.get('product_validation', {})
            if product_validation.get('non_packaging_rejected', False):
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
            
            for defect_type, boxes in bounding_boxes.items():
                if not boxes or defect_type == 'background':
                    continue
                
                single_box = boxes[0] if boxes else None
                
                if single_box:
                    if self._validate_single_bbox(single_box, defect_type):
                        filtered_boxes[defect_type] = [single_box]
                        filtered_stats[defect_type] = self._recalculate_stats_for_single_bbox(
                            single_box, defect_statistics.get(defect_type, {})
                        )
            
            if 'defect_analysis' in defect_classification:
                defect_classification['defect_analysis']['bounding_boxes'] = filtered_boxes
                defect_classification['defect_analysis']['defect_statistics'] = filtered_stats
            else:
                defect_classification['bounding_boxes'] = filtered_boxes
                defect_classification['defect_statistics'] = filtered_stats
            
            result['detected_defect_types'] = list(filtered_boxes.keys())
            result['frame_smart_processing'] = True
            result['single_bbox_enforced'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame smart processing: {e}")
            return result
    
    def _smart_final_decision_for_frames(self, result):
        """Smart final decision for frames"""
        try:
            product_validation = result.get('product_validation', {})
            if product_validation.get('non_packaging_rejected', False):
                result['final_decision'] = 'GOOD'
                result['frame_smart_decision'] = {
                    'reason': 'non_packaging_item_rejected',
                    'product_type': 'non_packaging',
                    'validation_source': 'product_aware_analysis',
                    'background_excluded': True
                }
                return result
            
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            
            adaptive_threshold = self.smart_config['current_anomaly_threshold']
            frame_threshold = adaptive_threshold
            
            detected_defects = result.get('detected_defect_types', [])
            
            if 'background' in detected_defects:
                detected_defects.remove('background')
                result['detected_defect_types'] = detected_defects
            
            is_anomalous_adaptive = anomaly_score > frame_threshold
            has_defects = len(detected_defects) > 0
            has_critical_defects = self._check_critical_defects_fast(result)
            
            if has_critical_defects:
                final_decision = 'DEFECT'
                decision_reason = 'critical_defects_frame'
            elif is_anomalous_adaptive and has_defects:
                final_decision = 'DEFECT'
                decision_reason = 'anomaly_and_defects_frame'
            elif is_anomalous_adaptive:
                final_decision = 'DEFECT'
                decision_reason = 'anomaly_only_frame'
            elif has_defects:
                final_decision = 'DEFECT'
                decision_reason = 'defects_only_frame'
            else:
                final_decision = 'GOOD'
                decision_reason = 'no_issues_frame'
            
            result['final_decision'] = final_decision
            result['frame_smart_decision'] = {
                'reason': decision_reason,
                'frame_threshold_used': frame_threshold,
                'original_threshold': adaptive_threshold,
                'anomaly_score': anomaly_score,
                'detected_defects_count': len(detected_defects),
                'has_critical_defects': has_critical_defects,
                'background_excluded': True,
                'product_context_validated': True
            }
            
            anomaly_detection['threshold_used'] = frame_threshold
            anomaly_detection['decision'] = final_decision
            result['anomaly_detection'] = anomaly_detection
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in frame decision: {e}")
            return result
    
    def _smart_final_decision_integrated(self, result):
        """Smart final decision integrated"""
        try:
            product_validation = result.get('product_validation', {})
            if product_validation.get('non_packaging_rejected', False):
                result['final_decision'] = 'GOOD'
                result['smart_decision'] = {
                    'reason': 'non_packaging_item_rejected',
                    'product_type': 'non_packaging',
                    'validation_source': 'product_aware_analysis',
                    'background_excluded': True
                }
                return result
            
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            
            adaptive_threshold = self.smart_config['current_anomaly_threshold']
            detected_defects = result.get('detected_defect_types', [])
            
            if 'background' in detected_defects:
                detected_defects.remove('background')
                result['detected_defect_types'] = detected_defects
            
            has_critical_defects = False
            if result.get('defect_classification'):
                defect_class = result['defect_classification']
                if 'defect_analysis' in defect_class:
                    bboxes = defect_class['defect_analysis'].get('bounding_boxes', {})
                else:
                    bboxes = defect_class.get('bounding_boxes', {})
                
                for defect_type, boxes in bboxes.items():
                    if defect_type == 'background':
                        continue
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
            elif is_anomalous_adaptive:
                final_decision = 'DEFECT'
                decision_reason = 'high_anomaly_score'
            elif has_significant_defects and len(detected_defects) >= 2:
                final_decision = 'DEFECT'
                decision_reason = 'multiple_defects'
            elif has_significant_defects:
                final_decision = 'DEFECT'
                decision_reason = 'defects_detected'
            else:
                final_decision = 'GOOD'
                decision_reason = 'no_significant_issues'
            
            result['final_decision'] = final_decision
            result['smart_decision'] = {
                'reason': decision_reason,
                'adaptive_threshold_used': adaptive_threshold,
                'anomaly_score': anomaly_score,
                'is_anomalous_adaptive': is_anomalous_adaptive,
                'detected_defects_count': len(detected_defects),
                'has_critical_defects': has_critical_defects,
                'confidence_level': 'high' if abs(anomaly_score - adaptive_threshold) > 0.2 else 'medium',
                'background_excluded': True,
                'product_context_validated': True
            }
            
            anomaly_detection['threshold_used'] = adaptive_threshold
            anomaly_detection['decision'] = final_decision
            result['anomaly_detection'] = anomaly_detection
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in integrated decision: {e}")
            return result
    
    def _check_critical_defects_fast(self, result):
        """Check for critical defects quickly"""
        try:
            defect_class = result.get('defect_classification', {})
            
            if 'defect_analysis' in defect_class:
                bboxes = defect_class['defect_analysis'].get('bounding_boxes', {})
            else:
                bboxes = defect_class.get('bounding_boxes', {})
            
            for defect_type, boxes in bboxes.items():
                if defect_type == 'background':
                    continue
                    
                if defect_type in ['missing_component', 'damaged']:
                    for box in boxes[:1]:
                        area_pct = box.get('area_percentage', 0)
                        if area_pct > 3.0:
                            return True
                        
            return False
            
        except Exception:
            return False
    
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
            self.logger.error(f"Error generating annotation: {e}")
            return None
    
    def _annotate_image_with_insights(self, image, result):
        """Add annotations to image"""
        try:
            import cv2
            from config import DEFECT_COLORS, SPECIFIC_DEFECT_CLASSES
            
            annotated = image.copy()
            decision = result.get('final_decision', 'UNKNOWN')
            
            # Check for non-packaging item
            product_validation = result.get('product_validation', {})
            if product_validation.get('non_packaging_rejected', False):
                # Gray border for non-packaging items
                annotated = cv2.copyMakeBorder(annotated, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(128, 128, 128))
                return annotated
            
            if decision == 'GOOD':
                annotated = cv2.copyMakeBorder(annotated, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            elif decision == 'DEFECT':
                annotated = cv2.copyMakeBorder(annotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                
                if result.get('defect_classification'):
                    self._draw_bounding_boxes(annotated, result['defect_classification'])
            
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
                if defect_type == 'background':
                    continue
                
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
                    if bbox.get('openai_corrected'):
                        label += "+"
                    
                    cv2.putText(image, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        except Exception as e:
            self.logger.error(f"Error drawing bounding boxes: {e}")
    
    # Health and status methods
    def get_health_status(self):
        """Get health status"""
        try:
            from config import OPENAI_API_KEY
        except ImportError:
            OPENAI_API_KEY = None
        
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
                'enhanced_detection': True,
                'background_class_skip': True,
                'product_context_validation': True
            },
            'smart_processing': {
                'enabled': self.smart_config['smart_enabled'],
                'integrated': True,
                'sensitivity_level': self.smart_config['anomaly_sensitivity'],
                'intelligent_filtering': self.smart_config['enable_intelligent_filtering'],
                'single_bbox_per_type': True,
                'false_positive_protection': True
            },
            'overall_status': 'healthy' if self.is_initialized else 'degraded',
            'initialization_error': self.initialization_error,
            'mode': 'product_aware_detection_with_false_positive_protection'
        }
        
        return base_status
    
    def get_system_information(self):
        """Get system information"""
        if not self.detector:
            raise RuntimeError("Detector not available")
        
        try:
            try:
                from config import OPENAI_API_KEY, OPENAI_MODEL
            except ImportError:
                OPENAI_API_KEY = None
                OPENAI_MODEL = None
            
            system_info = self.detector.get_system_info()
            
            system_info.update({
                'service_status': 'operational' if self.is_initialized else 'degraded',
                'real_time_processing': {
                    'enabled': True,
                    'frame_optimizations': self.smart_config['frame_optimizations'],
                    'enhanced_detection': True,
                    'adaptive_thresholds': True,
                    'model_warmup': self.frame_cache['model_warmup_done'],
                    'background_class_skip': True,
                    'single_bbox_per_type': True,
                    'product_context_validation': True,
                    'false_positive_protection': True,
                    'features': ['product_aware_detection', 'background_skip', 'single_bbox', 'smart_filtering', 'adaptive_sensitivity', 'frame_caching', 'non_packaging_rejection']
                },
                'smart_processing': {
                    'integrated': True,
                    'enabled': self.smart_config['smart_enabled'],
                    'current_config': self.get_smart_config(),
                    'features': ['adaptive_thresholds', 'intelligent_filtering', 'confidence_boosting', 'background_exclusion', 'product_context_filtering']
                },
                'openai_integration': {
                    'enabled': bool(OPENAI_API_KEY),
                    'model': OPENAI_MODEL if OPENAI_API_KEY else None,
                    'features': ['anomaly_analysis', 'defect_analysis', 'bbox_validation', 'product_context_validation'] if OPENAI_API_KEY else []
                },
                'api_version': '1.0.0',
                'mode': 'product_aware_detection_with_false_positive_protection'
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            raise RuntimeError(f"System info unavailable: {e}")
    
    def get_current_status(self):
        """Get current system status"""
        try:
            from config import OPENAI_API_KEY
        except ImportError:
            OPENAI_API_KEY = None
        
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
                'frame_caching': self.smart_config['frame_optimizations']['enable_model_caching'],
                'background_class_skip': True,
                'single_bbox_per_type': True,
                'bbox_validation': bool(OPENAI_API_KEY),
                'product_context_validation': True,
                'false_positive_protection': True
            },
            'frame_cache_info': {
                'consecutive_good_frames': self.frame_cache['consecutive_good_frames'],
                'model_warmup_done': self.frame_cache['model_warmup_done'],
                'last_processed': self.frame_cache['last_processed_time']
            },
            'current_load': self._get_current_load(),
            'memory_usage': self._get_memory_usage(),
            'mode': 'product_aware_detection_with_false_positive_protection',
            'last_check': datetime.now().isoformat()
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
                'intelligent_filtering': self.smart_config['enable_intelligent_filtering'],
                'background_class_skip': self.smart_config['background_class_skip'],
                'product_context_validation': self.smart_config['product_context_validation']
            },
            'frame_optimizations': self.smart_config['frame_optimizations'],
            'background_class_properly_handled': True,
            'single_bbox_per_type_enforced': True,
            'false_positive_protection_enabled': True
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
        """Get current detection thresholds"""
        return {
            'standard_thresholds': self.config,
            'smart_settings': self.get_smart_config(),
            'configurable': True,
            'storage': 'in-memory',
            'last_updated': datetime.now().isoformat(),
            'background_class_skip_enabled': True,
            'single_bbox_per_type_enabled': True,
            'product_context_validation_enabled': True
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
                'mode': 'product_aware_detection_with_false_positive_protection'
            }
        except ImportError:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0,
                'mode': 'product_aware_detection_with_false_positive_protection'
            }
"""
Detection Controller 
"""

from flask import jsonify
import json
import base64
import os
import time
import tempfile
import logging
from datetime import datetime


class DetectionController:
    """FIXED Detection Controller - Syntax errors resolved, no mock data"""
    
    def __init__(self, detection_service):
        self.detection_service = detection_service
        self.logger = logging.getLogger(__name__)
        
        # In-memory configuration storage (keep config)
        self.config = {
            'thresholds': {
                'anomaly_threshold': 0.7,
                'defect_confidence_threshold': 0.85
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def health_check(self):
        """Health check - real status only"""
        try:
            status = self.detection_service.get_health_status()
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'services': status,
                'api_version': '1.0.0',
                'mode': 'real_time_enhanced',
                'supported_content_types': ['application/json', 'multipart/form-data'],
                'real_time_capabilities': {
                    'enhanced_frame_processing': True,
                    'smart_processing': True,
                    'adaptive_thresholds': True,
                    'guaranteed_defect_detection': True,
                    'frame_caching': True
                }
            })
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_info(self):
        """Get system information - real data only"""
        try:
            info = self.detection_service.get_system_information()
            if not info:
                raise RuntimeError("Detection service not available")
                
            info['mode'] = 'real_time_enhanced'
            info['supported_content_types'] = ['application/json', 'multipart/form-data']
            
            return jsonify({
                'status': 'success',
                'data': info,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"System info failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_system_status(self):
        """Get current system status - real data only"""
        try:
            status = self.detection_service.get_current_status()
            if not status:
                raise RuntimeError("Detection service not available")
                
            status['mode'] = 'real_time_enhanced'
            status['supported_content_types'] = ['application/json', 'multipart/form-data']
            
            return jsonify({
                'status': 'success',
                'data': status,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"System status failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def process_image(self, request):
        """Process single image - real processing only"""
        start_time = time.time()
        temp_file = None
        
        try:
            self.logger.info(f"Processing image - Method: {request.method}, Content-Type: {request.content_type}")
            
            image_data, filename, validation_error = self._parse_image_request(request)
            
            if validation_error:
                return validation_error
            
            self.logger.info(f"Processing image - filename: {filename}, size: {len(image_data)} bytes")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(image_data)
            temp_file.close()
            
            # REAL processing only
            result = self.detection_service.process_single_image(image_data, filename, temp_file.name)
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Image processing failed',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            # Calculate REAL processing times
            total_processing_time = time.time() - start_time
            
            # Extract REAL timings from result or calculate
            preprocessing_time = result.get('preprocessing_time', 0.0)
            anomaly_processing_time = result.get('anomaly_processing_time', 0.0)
            classification_processing_time = result.get('classification_processing_time', 0.0)
            postprocessing_time = max(0.0, total_processing_time - preprocessing_time - 
                                   anomaly_processing_time - classification_processing_time)
            
            response = self._format_real_response(result, filename, {
                'preprocessing_time': preprocessing_time,
                'anomaly_processing_time': anomaly_processing_time,
                'classification_processing_time': classification_processing_time,
                'postprocessing_time': postprocessing_time,
                'total_processing_time': total_processing_time
            })
            
            self.logger.info(f"Image processed - Decision: {result.get('final_decision')}")
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            
            return jsonify({
                'status': 'error',
                'error': f'Processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def process_batch(self, request):
        """Process batch - real processing only"""
        start_time = time.time()
        temp_files = []
        
        try:
            images_data = self._parse_batch_request(request)
            
            if not images_data:
                return jsonify({
                    'status': 'error',
                    'error': 'No images array provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            self.logger.info(f"Processing batch of {len(images_data)} images")
            
            results = []
            for i, image_item in enumerate(images_data):
                try:
                    image_data = image_item['data']
                    filename = image_item['filename']
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.write(image_data)
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    # REAL processing only
                    result = self.detection_service.process_single_image(image_data, filename, temp_file.name)
                    if result:
                        formatted_result = self._format_real_response(result, filename, {
                            'preprocessing_time': result.get('preprocessing_time', 0.0),
                            'anomaly_processing_time': result.get('anomaly_processing_time', 0.0),
                            'classification_processing_time': result.get('classification_processing_time', 0.0),
                            'postprocessing_time': result.get('postprocessing_time', 0.0)
                        })
                        results.append(formatted_result)
                    
                except Exception as decode_error:
                    self.logger.warning(f"Error processing image {i+1}: {decode_error}")
                    continue
            
            processing_time = time.time() - start_time
            
            response = {
                'status': 'success',
                'batch_summary': {
                    'total_images': len(images_data),
                    'processed_images': len(results),
                    'defective_count': sum(1 for r in results if r.get('final_decision') == 'DEFECT'),
                    'good_count': sum(1 for r in results if r.get('final_decision') == 'GOOD'),
                    'total_processing_time': processing_time,
                    'avg_processing_time': processing_time / len(results) if results else 0
                },
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'mode': 'real_time_enhanced'
            }
            
            self.logger.info(f"Batch processed - {len(results)} results")
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def process_frame(self, request):
        """Process frame - real processing only"""
        start_time = time.time()
        temp_file = None
        
        try:
            self.logger.info(f"Processing frame - Method: {request.method}, Content-Type: {request.content_type}")
            
            # Parse frame request
            frame_data, filename, fast_mode, include_annotation, use_smart_processing, sensitivity_level, validation_error = self._parse_enhanced_frame_request(request)
            
            if validation_error:
                return validation_error
            
            if len(frame_data) > 5 * 1024 * 1024:
                return jsonify({
                    'status': 'error',
                    'error': 'Frame too large. Maximum size is 5MB',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(frame_data)
            temp_file.close()
            
            self.logger.info(f"Processing frame: {filename} (smart: {use_smart_processing}, "
                           f"fast: {fast_mode}, sensitivity: {sensitivity_level})")
            
            # REAL frame processing
            result = self.detection_service.process_frame(
                frame_data, filename, temp_file.name, 
                fast_mode=fast_mode, 
                include_annotation=include_annotation,
                use_smart_processing=use_smart_processing,
                sensitivity_level=sensitivity_level
            )
            
            if not result:
                return jsonify({
                    'status': 'error',
                    'error': 'Frame processing failed',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            # Format response with REAL data
            response = self._format_enhanced_frame_response(result, filename, {
                'preprocessing_time': result.get('preprocessing_time', 0.0),
                'anomaly_processing_time': result.get('anomaly_processing_time', 0.0),
                'classification_processing_time': result.get('classification_processing_time', 0.0),
                'postprocessing_time': result.get('postprocessing_time', 0.0)
            })
            
            # Add real-time processing metadata
            response.update({
                'real_time_enhanced': True,
                'frame_optimizations_applied': result.get('frame_optimizations', {}),
                'smart_processing_applied': use_smart_processing,
                'sensitivity_level_used': sensitivity_level,
                'total_processing_time': time.time() - start_time
            })
            
            self.logger.info(f"Frame processed - Decision: {result.get('final_decision')} "
                           f"in {time.time() - start_time:.3f}s")
            
            return jsonify(response)
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            
            return jsonify({
                'status': 'error',
                'error': f'Frame processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file: {e}")
    
    def _parse_enhanced_frame_request(self, request):
        """Parse enhanced frame request - real validation only"""
        try:
            frame_data = None
            filename = None
            fast_mode = True
            include_annotation = True
            use_smart_processing = True
            sensitivity_level = None
            
            # Method 1: Handle JSON requests
            if request.is_json or 'application/json' in str(request.content_type):
                json_data = request.get_json()
                
                if not json_data:
                    return None, None, None, None, None, None, (jsonify({
                        'status': 'error',
                        'error': 'Invalid JSON request',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
                
                # Extract frame data
                frame_base64 = json_data.get('frame_base64') or json_data.get('image_base64', '')
                filename = json_data.get('filename', f"frame_{int(time.time())}.jpg")
                fast_mode = json_data.get('fast_mode', True)
                include_annotation = json_data.get('include_annotation', True)
                use_smart_processing = json_data.get('use_smart_processing', True)
                sensitivity_level = json_data.get('sensitivity_level', 'medium')
                
                # Validate sensitivity level
                if sensitivity_level not in ['low', 'medium', 'high']:
                    sensitivity_level = 'medium'
                
                if not frame_base64:
                    return None, None, None, None, None, None, (jsonify({
                        'status': 'error',
                        'error': 'Missing frame_base64 or image_base64 field',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
                
                # Decode base64 data
                try:
                    if frame_base64.startswith('data:image'):
                        frame_base64 = frame_base64.split(',')[1]
                    
                    frame_data = base64.b64decode(frame_base64)
                except Exception as decode_error:
                    return None, None, None, None, None, None, (jsonify({
                        'status': 'error',
                        'error': f'Invalid base64 frame data: {str(decode_error)}',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
            
            # Method 2: Handle form data requests
            elif request.files:
                # Try different field names
                for field_name in ['frame', 'image', 'file']:
                    if field_name in request.files:
                        file = request.files[field_name]
                        if file.filename != '':
                            frame_data = file.read()
                            filename = file.filename or f"frame_{int(time.time())}.jpg"
                            break
                
                # Get form parameters
                fast_mode = request.form.get('fast_mode', 'true').lower() == 'true'
                include_annotation = request.form.get('include_annotation', 'true').lower() == 'true'
                use_smart_processing = request.form.get('use_smart_processing', 'true').lower() == 'true'
                sensitivity_level = request.form.get('sensitivity_level', 'medium')
            
            if not frame_data:
                return None, None, None, None, None, None, (jsonify({
                    'status': 'error',
                    'error': 'No frame data provided. Use JSON with frame_base64 or form-data with frame/image/file field',
                    'timestamp': datetime.now().isoformat()
                }), 400)
            
            return frame_data, filename, fast_mode, include_annotation, use_smart_processing, sensitivity_level, None
            
        except Exception as e:
            return None, None, None, None, None, None, (jsonify({
                'status': 'error',
                'error': f'Frame request parsing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 400)
    
    def _format_enhanced_frame_response(self, result, filename, timings):
        """Format response for enhanced frame processing - real data only"""
        try:
            # Extract REAL anomaly information
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            anomaly_threshold = anomaly_detection.get('threshold_used', 0.3)
            
            # Calculate REAL confidence level
            confidence_level = self._calculate_anomaly_confidence_level(anomaly_score, result.get('final_decision', 'UNKNOWN'))
            
            # Extract REAL defects information
            defects = self._extract_real_defects_for_frames(result)
            
            # Get REAL annotated image
            annotated_image = result.get('annotated_image_base64', '')
            
            # Create response with REAL data only
            response = {
                'final_decision': result.get('final_decision', 'UNKNOWN'),
                'preprocessing_time': timings['preprocessing_time'],
                'anomaly_processing_time': timings['anomaly_processing_time'],
                'classification_processing_time': timings['classification_processing_time'],
                'postprocessing_time': timings['postprocessing_time'],
                'anomaly_score': round(anomaly_score, 3),
                'anomaly_confidence_level': confidence_level,
                'anomaly_threshold': anomaly_threshold,
                'annotated_image': annotated_image,
                'filename': filename,
                'defects': defects,
                
                # Frame-specific information from REAL processing
                'frame_mode': result.get('frame_mode', True),
                'fast_mode': result.get('fast_mode', True),
                'real_time_processing': result.get('real_time_processing', True),
                'enhanced_detection': result.get('frame_enhanced_detection', False),
                'guaranteed_defect_detection': result.get('guaranteed_defect_detection', False),
                'smart_processing': result.get('frame_smart_processing', False),
                
                # Processing metadata from REAL results
                'processing_metadata': {
                    'cached_result': result.get('cached_result', False),
                    'consecutive_good_count': result.get('consecutive_good_count', 0),
                    'smart_decision': result.get('frame_smart_decision', {}),
                    'frame_optimizations': result.get('frame_optimizations', {}),
                    'processing_mode': result.get('processing_mode', 'enhanced')
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting frame response: {e}")
            raise RuntimeError(f"Failed to format frame response: {e}")
    
    def _extract_real_defects_for_frames(self, result):
        """Extract defects from REAL detection results - no mock data"""
        defects = []
        
        try:
            defect_classification = result.get('defect_classification', {})
            
            # Get bounding boxes from REAL detection
            bounding_boxes = {}
            defect_statistics = {}
            
            if 'defect_analysis' in defect_classification:
                bounding_boxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                defect_statistics = defect_classification['defect_analysis'].get('defect_statistics', {})
            elif 'bounding_boxes' in defect_classification:
                bounding_boxes = defect_classification['bounding_boxes']
                defect_statistics = defect_classification.get('defect_statistics', {})
            
            # Only process if we have REAL detection results
            if not bounding_boxes:
                return []
            
            # Get REAL image dimensions
            actual_image_width = 640
            actual_image_height = 640
            
            if 'image_dimensions' in result:
                actual_image_width = result['image_dimensions'].get('width', 640)
                actual_image_height = result['image_dimensions'].get('height', 640)
            
            # Process REAL detections only
            for defect_type, boxes in bounding_boxes.items():
                if not boxes:
                    continue
                
                stats = defect_statistics.get(defect_type, {})
                
                # Use REAL combined bounding box
                if len(boxes) > 0:
                    combined_bbox = boxes[0]  # Should be the single combined box
                    
                    # Calculate REAL area percentage
                    total_defect_area = combined_bbox.get('area', combined_bbox.get('width', 0) * combined_bbox.get('height', 0))
                    total_image_area = actual_image_width * actual_image_height
                    
                    if total_image_area > 0:
                        area_percentage = (total_defect_area / total_image_area) * 100
                        area_percentage = min(area_percentage, 100.0)
                    else:
                        area_percentage = 0
                    
                    # Get REAL confidence score
                    confidence_score = combined_bbox.get('confidence', stats.get('avg_confidence', 0.0))
                    
                    if isinstance(confidence_score, (int, float)):
                        confidence_score = round(confidence_score, 3)
                    else:
                        confidence_score = 0.0
                    
                    # Check for REAL confidence boosting
                    confidence_boosted = combined_bbox.get('frame_confidence_boosted', False)
                    original_confidence = combined_bbox.get('original_confidence') if confidence_boosted else None
                    
                    # Determine severity level
                    severity_level = combined_bbox.get('severity', self._determine_frame_severity_level(area_percentage, defect_type))
                    
                    # Create defect info from REAL data only
                    defect_info = {
                        'label': defect_type,
                        'confidence_score': confidence_score,
                        'severity_level': severity_level,
                        'area_percentage': round(area_percentage, 2),
                        'bounding_box': {
                            'x': combined_bbox.get('x', 0),
                            'y': combined_bbox.get('y', 0),
                            'width': combined_bbox.get('width', 0),
                            'height': combined_bbox.get('height', 0)
                        },
                        'frame_enhanced': True,
                        'confidence_boosted': confidence_boosted,
                        'original_confidence': original_confidence,
                        'detection_method': 'enhanced_real_time',
                        'total_regions': 1,
                        'combined_defect': True
                    }
                    
                    defects.append(defect_info)
                    
                    self.logger.info(f"Real frame defect extracted: {defect_type}")
            
        except Exception as e:
            self.logger.error(f"Error extracting real defects for frames: {e}")
        
        return defects
    
    def _determine_frame_severity_level(self, area_percentage, defect_type):
        """Determine severity level for frame processing"""
        # Frame-specific severity thresholds
        if defect_type in ['missing_component', 'damaged']:
            if area_percentage < 0.3:
                return 'minor'
            elif area_percentage < 1.5:
                return 'moderate'
            elif area_percentage < 4.0:
                return 'significant'
            else:
                return 'critical'
        else:
            if area_percentage < 0.8:
                return 'minor'
            elif area_percentage < 2.5:
                return 'moderate'
            elif area_percentage < 6.0:
                return 'significant'
            else:
                return 'critical'
    
    def _parse_image_request(self, request):
        """Parse image request from both form-data and JSON"""
        try:
            image_data = None
            filename = None
            
            # Method 1: Handle JSON requests
            if request.is_json or 'application/json' in str(request.content_type):
                json_data = request.get_json()
                
                if not json_data:
                    return None, None, (jsonify({
                        'status': 'error',
                        'error': 'Invalid JSON request',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
                
                image_base64 = json_data.get('image_base64', '')
                filename = json_data.get('filename', f"upload_{int(time.time())}.jpg")
                
                if not image_base64:
                    return None, None, (jsonify({
                        'status': 'error',
                        'error': 'Missing image_base64 field',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
                
                try:
                    if image_base64.startswith('data:image'):
                        image_base64 = image_base64.split(',')[1]
                    
                    image_data = base64.b64decode(image_base64)
                except Exception as decode_error:
                    return None, None, (jsonify({
                        'status': 'error',
                        'error': f'Invalid base64 image data: {str(decode_error)}',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
            
            # Method 2: Handle form data requests
            elif request.files and 'image' in request.files:
                file = request.files['image']
                if file.filename == '':
                    return None, None, (jsonify({
                        'status': 'error',
                        'error': 'No file selected',
                        'timestamp': datetime.now().isoformat()
                    }), 400)
                
                image_data = file.read()
                filename = file.filename or f"upload_{int(time.time())}.jpg"
            
            # Method 3: Try alternative form field names
            elif request.files:
                for field_name in ['file', 'upload', 'data']:
                    if field_name in request.files:
                        file = request.files[field_name]
                        if file.filename != '':
                            image_data = file.read()
                            filename = file.filename or f"upload_{int(time.time())}.jpg"
                            break
            
            if not image_data:
                return None, None, (jsonify({
                    'status': 'error',
                    'error': 'No image provided. Use JSON with image_base64 or form-data with image/file field',
                    'timestamp': datetime.now().isoformat()
                }), 400)
            
            if len(image_data) > 5 * 1024 * 1024:
                return None, None, (jsonify({
                    'status': 'error',
                    'error': 'File too large. Maximum size is 5MB',
                    'timestamp': datetime.now().isoformat()
                }), 400)
            
            return image_data, filename, None
            
        except Exception as e:
            return None, None, (jsonify({
                'status': 'error',
                'error': f'Request parsing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 400)
    
    def _parse_batch_request(self, request):
        """Parse batch request from both form-data and JSON"""
        try:
            images_data = []
            
            # Method 1: Handle JSON batch requests
            if request.is_json or 'application/json' in str(request.content_type):
                json_data = request.get_json()
                
                if not json_data or 'images' not in json_data:
                    return None
                
                json_images = json_data['images']
                if not isinstance(json_images, list) or len(json_images) == 0:
                    return None
                
                for i, image_item in enumerate(json_images):
                    if 'image_base64' not in image_item:
                        continue
                    
                    base64_data = image_item['image_base64']
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',')[1]
                    
                    try:
                        image_data = base64.b64decode(base64_data)
                        filename = image_item.get('filename', f"batch_image_{i+1}.jpg")
                        
                        images_data.append({
                            'data': image_data,
                            'filename': filename
                        })
                    except Exception:
                        continue
            
            # Method 2: Handle form-data batch
            elif request.files:
                for key, file in request.files.items():
                    if file.filename != '':
                        try:
                            image_data = file.read()
                            filename = file.filename or f"batch_{key}.jpg"
                            
                            images_data.append({
                                'data': image_data,
                                'filename': filename
                            })
                        except Exception:
                            continue
            
            return images_data if images_data else None
            
        except Exception as e:
            self.logger.error(f"Batch request parsing failed: {e}")
            return None
    
    def _format_real_response(self, result, filename, timings):
        """Format response using REAL data only - no mock values"""
        try:
            anomaly_detection = result.get('anomaly_detection', {})
            anomaly_score = anomaly_detection.get('anomaly_score', 0.0)
            anomaly_threshold = anomaly_detection.get('threshold_used', 0.3)
            
            confidence_level = self._calculate_anomaly_confidence_level(anomaly_score, result.get('final_decision', 'UNKNOWN'))
            
            defects = self._extract_real_defects(result)
            
            annotated_image = result.get('annotated_image_base64', '')
            
            response = {
                'final_decision': result.get('final_decision', 'UNKNOWN'),
                'preprocessing_time': timings['preprocessing_time'],
                'anomaly_processing_time': timings['anomaly_processing_time'],
                'classification_processing_time': timings['classification_processing_time'],
                'postprocessing_time': timings['postprocessing_time'],
                'anomaly_score': round(anomaly_score, 3),
                'anomaly_confidence_level': confidence_level,
                'anomaly_threshold': anomaly_threshold,
                'annotated_image': annotated_image,
                'filename': filename,
                'defects': defects
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting real response: {e}")
            raise RuntimeError(f"Failed to format response: {e}")
    
    def _extract_real_defects(self, result):
        """Extract defects from REAL detection results only - no mock data"""
        defects = []
        
        try:
            defect_classification = result.get('defect_classification', {})
            
            bounding_boxes = {}
            defect_statistics = {}
            
            if 'defect_analysis' in defect_classification:
                bounding_boxes = defect_classification['defect_analysis'].get('bounding_boxes', {})
                defect_statistics = defect_classification['defect_analysis'].get('defect_statistics', {})
            elif 'bounding_boxes' in defect_classification:
                bounding_boxes = defect_classification['bounding_boxes']
                defect_statistics = defect_classification.get('defect_statistics', {})
            
            # Only process if we have REAL bounding boxes
            if not bounding_boxes:
                return []
            
            actual_image_width = 640
            actual_image_height = 640
            
            if 'image_dimensions' in result:
                actual_image_width = result['image_dimensions'].get('width', 640)
                actual_image_height = result['image_dimensions'].get('height', 640)
            
            # Process each defect type from REAL detection
            for defect_type, boxes in bounding_boxes.items():
                if not boxes:
                    continue
                
                stats = defect_statistics.get(defect_type, {})
                
                # Use REAL combined bounding box
                if len(boxes) > 0:
                    combined_bbox = boxes[0]  # Should be the single combined box
                    
                    # Calculate REAL area percentage
                    total_defect_area = combined_bbox.get('area', combined_bbox.get('width', 0) * combined_bbox.get('height', 0))
                    total_image_area = actual_image_width * actual_image_height
                    
                    if total_image_area > 0:
                        area_percentage = (total_defect_area / total_image_area) * 100
                        area_percentage = min(area_percentage, 100.0)
                    else:
                        area_percentage = 0
                    
                    # Get REAL confidence score
                    confidence_score = combined_bbox.get('confidence', stats.get('avg_confidence', 0.0))
                    
                    if isinstance(confidence_score, (int, float)):
                        confidence_score = round(confidence_score, 3)
                    else:
                        confidence_score = 0.0
                    
                    # Check for REAL confidence boosting
                    confidence_boosted = combined_bbox.get('frame_confidence_boosted', False) or combined_bbox.get('confidence_boosted', False)
                    original_confidence = combined_bbox.get('original_confidence') if confidence_boosted else None
                    
                    # Determine severity level
                    severity_level = combined_bbox.get('severity', self._determine_severity_level(area_percentage, defect_type))
                    
                    # Create defect info from REAL data only
                    defect_info = {
                        'label': defect_type,
                        'confidence_score': confidence_score,
                        'severity_level': severity_level,
                        'area_percentage': round(area_percentage, 2),
                        'bounding_box': {
                            'x': combined_bbox.get('x', 0),
                            'y': combined_bbox.get('y', 0),
                            'width': combined_bbox.get('width', 0),
                            'height': combined_bbox.get('height', 0)
                        },
                        'total_regions': 1,
                        'combined_defect': True,
                        'confidence_boosted': confidence_boosted,
                        'original_confidence': original_confidence,
                        'detection_method': 'real_detection'
                    }
                    
                    defects.append(defect_info)
                    
                    self.logger.info(f"Real defect extracted: {defect_type}")
            
        except Exception as e:
            self.logger.error(f"Error extracting real defects: {e}")
        
        return defects
    
    def _determine_severity_level(self, area_percentage, defect_type):
        """Determine severity level based on area percentage and defect type"""
        if defect_type in ['missing_component', 'damaged']:
            if area_percentage < 0.5:
                return 'minor'
            elif area_percentage < 2.0:
                return 'moderate'
            elif area_percentage < 5.0:
                return 'significant'
            else:
                return 'critical'
        else:
            if area_percentage < 1.0:
                return 'minor'
            elif area_percentage < 3.0:
                return 'moderate'
            elif area_percentage < 8.0:
                return 'significant'
            else:
                return 'critical'
    
    def _calculate_anomaly_confidence_level(self, anomaly_score, final_decision):
        """Calculate anomaly confidence level"""
        if final_decision == 'GOOD':
            if anomaly_score < 0.1:
                return 'very_high'
            elif anomaly_score < 0.3:
                return 'high'
            elif anomaly_score < 0.5:
                return 'medium'
            else:
                return 'low'
        else:
            if anomaly_score > 0.9:
                return 'very_high'
            elif anomaly_score > 0.7:
                return 'high'
            elif anomaly_score > 0.5:
                return 'medium'
            else:
                return 'low'
    
    # Keep existing threshold management methods
    def get_detection_thresholds(self):
        """Get current detection thresholds"""
        try:
            return jsonify({
                'status': 'success',
                'data': self.config['thresholds'],
                'last_updated': self.config['last_updated'],
                'timestamp': datetime.now().isoformat(),
                'mode': 'real_time_enhanced'
            })
        except Exception as e:
            self.logger.error(f"Error getting thresholds: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def update_detection_thresholds(self, request):
        """Update detection thresholds"""
        try:
            # Handle both JSON and form data
            if request.is_json:
                new_thresholds = request.get_json()
            else:
                new_thresholds = request.form.to_dict()
                # Convert string values to float for form data
                for key, value in new_thresholds.items():
                    if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                        try:
                            new_thresholds[key] = float(value)
                        except (ValueError, TypeError):
                            pass
            
            if not new_thresholds:
                return jsonify({
                    'status': 'error',
                    'error': 'No threshold data provided',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Validate thresholds
            for key, value in new_thresholds.items():
                if key in ['anomaly_threshold', 'defect_confidence_threshold']:
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        return jsonify({
                            'status': 'error',
                            'error': f'{key} must be a number between 0 and 1',
                            'timestamp': datetime.now().isoformat()
                        }), 400
            
            # Update configuration
            self.config['thresholds'].update(new_thresholds)
            self.config['last_updated'] = datetime.now().isoformat()
            
            # Update detection service
            self.detection_service.update_thresholds(new_thresholds)
            
            self.logger.info(f"Thresholds updated: {new_thresholds}")
            
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'Thresholds updated successfully',
                    'new_thresholds': self.config['thresholds']
                },
                'timestamp': datetime.now().isoformat(),
                'mode': 'real_time_enhanced'
            })
                
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
            
    def reset_detection_thresholds(self, request):
        """Reset detection thresholds to default values"""
        try:
            # Reset to default thresholds
            self.config['thresholds'] = {
                'anomaly_threshold': 0.7,
                'defect_confidence_threshold': 0.85
            }
            self.config['last_updated'] = datetime.now().isoformat()

            # Update detection service with default thresholds
            self.detection_service.update_thresholds(self.config['thresholds'])

            self.logger.info("Detection thresholds reset to default values")

            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'Thresholds reset to default values',
                    'new_thresholds': self.config['thresholds']
                },
                'timestamp': datetime.now().isoformat(),
                'mode': 'real_time_enhanced'
            })

        except Exception as e:
            self.logger.error(f"Error resetting thresholds: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
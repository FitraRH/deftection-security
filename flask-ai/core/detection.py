# core/detection.py - Enhanced with OpenAI Bounding Box Validation and Type Correction
"""
Core detection logic with OpenAI analysis integration (OpenAI 1.x compatible)
ENHANCED with RAG prompts for accurate defect classification and bounding box validation
FIXED: OpenAI validation for bounding box accuracy and defect type correction
"""

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from openai import OpenAI
import base64
import io
from PIL import Image
from config import *


class DetectionCore:
    """Core detection functionality with enhanced OpenAI integration and bounding box validation"""
    
    def __init__(self, anomalib_model, hrnet_model, device='cuda'):
        self.anomalib_model = anomalib_model
        self.hrnet_model = hrnet_model
        self.device = device
        
        # Setup OpenAI 1.x client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_enabled = True
            print("OpenAI client initialized with bounding box validation and type correction")
        else:
            self.openai_client = None
            self.openai_enabled = False
            print("Warning: OpenAI API key not found")
        
        # Preprocessing for HRNet
        self.hrnet_transform = A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ToTensorV2()
        ])
    
    def detect_anomaly(self, image_path):
        """
        Layer 1: Anomaly detection with enhanced OpenAI analysis
        """
        if not self.anomalib_model:
            raise ValueError("Anomalib model not loaded")
        
        try:
            # Run Anomalib inference
            result = self.anomalib_model.predict(image=image_path)
            
            # Process anomaly results
            if isinstance(result.pred_score, torch.Tensor):
                anomaly_score = float(result.pred_score.cpu().item())
            else:
                anomaly_score = float(result.pred_score)
                
            if isinstance(result.pred_label, torch.Tensor):
                is_anomalous = bool(result.pred_label.cpu().item())
            else:
                is_anomalous = bool(result.pred_label)
            
            # Get anomaly mask if available
            anomaly_mask = None
            if hasattr(result, 'pred_mask') and result.pred_mask is not None:
                if isinstance(result.pred_mask, torch.Tensor):
                    anomaly_mask = result.pred_mask.cpu().numpy()
                else:
                    anomaly_mask = result.pred_mask
                
                if len(anomaly_mask.shape) > 2:
                    anomaly_mask = anomaly_mask[0]
            
            base_result = {
                'is_anomalous': is_anomalous,
                'anomaly_score': anomaly_score,
                'anomaly_mask': anomaly_mask,
                'threshold_used': ANOMALY_THRESHOLD,
                'decision': 'DEFECT' if (is_anomalous and anomaly_score > ANOMALY_THRESHOLD) else 'GOOD'
            }
            
            # Enhanced OpenAI Layer 1 Analysis
            if self.openai_enabled:
                print(f"Running enhanced OpenAI anomaly analysis (score: {anomaly_score:.3f})")
                openai_analysis = self._analyze_anomaly_with_openai_enhanced(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
            
            return base_result
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Layer 2: Defect classification with enhanced OpenAI analysis and bounding box validation
        """
        if not self.hrnet_model:
            raise ValueError("HRNet model not loaded")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image_rgb.shape[:2]
            
            # Apply region mask if provided
            if region_mask is not None:
                region_mask = cv2.resize(region_mask, (original_size[1], original_size[0]))
                masked_image = image_rgb.copy()
                masked_image[region_mask < 0.5] = 0
                image_rgb = masked_image
            
            # Preprocess for HRNet
            transformed = self.hrnet_transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # HRNet inference
            with torch.no_grad():
                output = self.hrnet_model(input_tensor)
                predictions = torch.softmax(output, dim=1)
                predicted_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
                confidence_scores = torch.max(predictions, dim=1)[0].squeeze().cpu().numpy()
            
            # Resize predictions to original size
            predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            confidence_scores = cv2.resize(confidence_scores, 
                                         (original_size[1], original_size[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            
            # FIXED: Use enhanced defect analysis with background class skip
            from core.enhanced_detection import analyze_defect_predictions_enhanced
            defect_analysis = analyze_defect_predictions_enhanced(predicted_mask, confidence_scores, original_size)
            
            base_result = {
                'predicted_mask': predicted_mask,
                'confidence_scores': confidence_scores,
                'defect_analysis': defect_analysis,
                'detected_defects': defect_analysis['detected_defects'],
                'bounding_boxes': defect_analysis['bounding_boxes'],
                'class_distribution': defect_analysis['class_distribution']
            }
            
            # Enhanced OpenAI Layer 2 Analysis with bounding box validation and type correction
            if self.openai_enabled and defect_analysis['detected_defects']:
                print(f"Running enhanced OpenAI defect analysis with bounding box validation and type correction")
                openai_analysis = self._analyze_defects_with_openai_enhanced(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
                
                # Apply OpenAI corrections if available
                if openai_analysis.get('bbox_corrections') or openai_analysis.get('type_corrections'):
                    corrections = {
                        'bbox_corrections': openai_analysis.get('bbox_corrections', {}),
                        'type_corrections': openai_analysis.get('type_corrections', {})
                    }
                    base_result = self._apply_openai_corrections(base_result, corrections)
            
            return base_result
            
        except Exception as e:
            print(f"Error in defect classification: {e}")
            return None
    
    def _analyze_anomaly_with_openai_enhanced(self, image_path, anomaly_result):
        """Enhanced OpenAI analysis for Layer 1 (Anomaly Detection) with RAG prompts"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            
            # Enhanced RAG prompt for anomaly detection
            prompt = f"""Analyze this product packaging image for quality control defects.

ANOMALY DETECTION MODEL RESULTS:
- Anomaly Score: {anomaly_result['anomaly_score']:.3f}
- Decision: {anomaly_result['decision']}
- Threshold: {anomaly_result['threshold_used']}

PACKAGING DEFECT TYPES TO IDENTIFY:

1. DAMAGED: Physical structural damage
   - Visual signs: Crushed areas, dented corners, collapsed sections, bent/warped material
   - Examples: Crushed cereal box corner, dented can, flattened plastic bottle
   - Severity indicators: Size of deformed area, depth of damage

2. MISSING_COMPONENT: Absent packaging elements
   - Visual signs: Missing caps/lids, absent labels/stickers, missing protective seals
   - Examples: Missing bottle cap, absent safety seal, missing product label
   - Severity indicators: Essential vs non-essential component

3. OPEN: Unwanted openings compromising closure
   - Visual signs: Holes showing dark interior, tears, rips, gaps in seams, punctures
   - Examples: Hole in plastic bag, torn cardboard flap, ripped food packaging
   - Severity indicators: Size of opening, contamination risk

4. SCRATCH: Surface abrasions affecting appearance
   - Visual signs: Thin linear marks, scrape marks, surface abrasions, scuff marks
   - Examples: Scratched plastic container, scuffed box surface, abraded label
   - Severity indicators: Depth of scratch, visibility

5. STAINED: Discoloration or contamination marks
   - Visual signs: Dark spots, discolored areas, dirty marks, water stains, grease marks
   - Examples: Water-stained cardboard, grease marks on packaging, dirt smudges
   - Severity indicators: Stain size, color contrast

ANALYSIS REQUIREMENTS:
1. Visual quality assessment based on defect types above
2. Confidence in anomaly detection accuracy (0-100%)
3. Key observations about packaging condition
4. Technical recommendation for quality control

Focus on accuracy using the specific defect classifications provided. Be precise and technical."""

            print("Calling OpenAI API with enhanced RAG prompt...")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            
            print(f"Enhanced OpenAI anomaly analysis completed - confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'model_used': OPENAI_MODEL,
                'layer': 'anomaly_detection',
                'rag_enhanced': True
            }
            
        except Exception as e:
            print(f"Enhanced OpenAI anomaly analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'error': str(e)
            }
    
    def _analyze_defects_with_openai_enhanced(self, image_path, defect_result):
        """Enhanced OpenAI analysis for Layer 2 with bounding box validation and defect type correction"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            image_base64 = self._encode_image_to_base64(image_path)
            
            detected_defects = defect_result['detected_defects']
            bounding_boxes = defect_result.get('bounding_boxes', {})
            
            # Create detailed bounding box information for validation
            bbox_info = ""
            total_bboxes = 0
            for defect_type, boxes in bounding_boxes.items():
                bbox_info += f"\n{defect_type.upper()}: {len(boxes)} regions detected"
                total_bboxes += len(boxes)
                for i, bbox in enumerate(boxes):
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    area_pct = bbox.get('area_percentage', 0)
                    conf = bbox.get('confidence', 0)
                    bbox_info += f"\n  Region {i+1}: Box({x},{y},{w},{h}) Area({area_pct:.1f}%) Conf({conf:.3f})"
            
            # Enhanced RAG prompt with bounding box validation focus
            prompt = f"""Analyze this product packaging image for defect classification accuracy and correction.

MODEL DETECTION RESULTS:
DETECTED DEFECTS: {', '.join(detected_defects) if detected_defects else 'None detected'}
TOTAL BOUNDING BOXES: {total_bboxes}
BOUNDING BOX DETAILS:{bbox_info if bbox_info else ' None provided'}

CRITICAL VISUAL ANALYSIS TASKS:

1. DEFECT TYPE CORRECTION: Look at the actual visual defects in the image
   - OPEN: Holes, tears, gaps, punctures showing dark interior or background
   - SCRATCH: Linear surface marks, abrasions, scuff marks on surface
   - MISSING_COMPONENT: Absent parts like caps, labels, seals
   - DAMAGED: Physical structural damage like crushed, dented areas
   - STAINED: Discoloration, spots, contamination marks

2. COMMON MISCLASSIFICATION PATTERNS:
   - Open holes often misclassified as "missing_component" or "stained"
   - Surface scratches misclassified as "stained"
   - Large background areas incorrectly detected as "defects"

3. WHAT DO YOU ACTUALLY SEE in this image?
   - Describe the visible defects in plain language
   - Are there holes, tears, or openings? (This would be "open")
   - Are there surface scratches or marks? (This would be "scratch")
   - Are there missing parts? (This would be "missing_component")

4. BOUNDING BOX VALIDATION:
   - Are boxes positioned on actual defects or empty background?
   - Do boxes cover 50%+ of image? (Likely false positive)
   - Are coordinates reasonable for the defect type?

5. PROVIDE CORRECTIONS:
   If you see defects that are misclassified, specify:
   CORRECT_TYPE: [actual_defect_type] - [reason why this is the correct type]
   
   If bounding boxes are wrong, specify:
   BBOX_CORRECTION: [defect_type]: x,y,width,height - [reason for correction]

Focus on what you ACTUALLY see in the image vs what the model detected. The model may have incorrectly classified an "open" defect as something else."""

            print("Calling OpenAI API for enhanced defect and bounding box validation...")
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            bbox_confidence = self._extract_bbox_confidence(analysis_text)
            
            # Extract corrections from OpenAI response
            corrections = self._extract_bbox_corrections(analysis_text)
            
            print(f"Enhanced OpenAI defect analysis completed - confidence: {confidence}%, bbox: {bbox_confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'bbox_validation': {
                    'confidence': bbox_confidence,
                    'validated_regions': len(bounding_boxes),
                    'spatial_accuracy': 'high' if bbox_confidence > 80 else 'medium' if bbox_confidence > 60 else 'low'
                },
                'bbox_corrections': corrections.get('bbox_corrections', {}),
                'type_corrections': corrections.get('type_corrections', {}),
                'model_used': OPENAI_MODEL,
                'layer': 'defect_classification',
                'defects_analyzed': detected_defects,
                'bounding_boxes_analyzed': {k: len(v) for k, v in bounding_boxes.items()},
                'rag_enhanced': True,
                'classification_validation': True,
                'spatial_validation': True,
                'type_correction_enabled': True
            }
            
        except Exception as e:
            print(f"Enhanced OpenAI defect analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'bbox_validation': {'confidence': 0, 'error': str(e)},
                'error': str(e)
            }
    
    def _extract_bbox_corrections(self, analysis_text):
        """Extract bounding box corrections and defect type corrections from OpenAI analysis"""
        corrections = {}
        type_corrections = {}
        
        try:
            import re
            
            # Extract defect type corrections
            type_patterns = [
                r'CORRECT_TYPE:\s*(\w+)\s*-\s*([^\n]+)',
                r'should be\s+(\w+)\s+because\s+([^\n]+)',
                r'actually\s+(\w+)\s+defect\s+([^\n]*)',
                r'correct\s+type\s+is\s+(\w+)\s+([^\n]*)'
            ]
            
            for pattern in type_patterns:
                matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        defect_type = groups[0].lower()
                        reason = groups[1] if len(groups) > 1 else "OpenAI correction"
                        
                        # Validate defect type
                        valid_types = ['open', 'scratch', 'missing_component', 'damaged', 'stained']
                        if defect_type in valid_types:
                            type_corrections[defect_type] = {
                                'corrected_type': defect_type,
                                'reason': reason,
                                'source': 'openai_type_correction'
                            }
            
            # Extract bounding box corrections
            bbox_patterns = [
                r'BBOX_CORRECTION:\s*(\w+):\s*(\d+),(\d+),(\d+),(\d+)\s*-\s*([^\n]+)',
                r'CORRECTION:\s*(\w+):\s*(\d+),(\d+),(\d+),(\d+)\s*\(([^)]+)\)',
                r'(\w+)\s+box\s+should\s+be\s+at\s+(\d+),(\d+)\s+size\s+(\d+)x(\d+)',
                r'move\s+(\w+)\s+to\s+(\d+),(\d+),(\d+),(\d+)'
            ]
            
            for pattern in bbox_patterns:
                matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 5:
                        defect_type = groups[0].lower()
                        x, y = int(groups[1]), int(groups[2])
                        w, h = int(groups[3]), int(groups[4])
                        reason = groups[5] if len(groups) > 5 else "OpenAI bbox correction"
                        
                        corrections[defect_type] = {
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'reason': reason,
                            'source': 'openai_validation'
                        }
            
            # Return both types of corrections
            return {
                'bbox_corrections': corrections,
                'type_corrections': type_corrections
            }
            
        except Exception as e:
            print(f"Error extracting corrections: {e}")
            return {'bbox_corrections': {}, 'type_corrections': {}}
    
    def _apply_openai_corrections(self, result, corrections):
        """Apply OpenAI bounding box and type corrections to detection result"""
        try:
            if not corrections:
                return result
            
            bbox_corrections = corrections.get('bbox_corrections', {})
            type_corrections = corrections.get('type_corrections', {})
            
            print(f"Applying {len(bbox_corrections)} bbox corrections and {len(type_corrections)} type corrections...")
            
            bounding_boxes = result.get('bounding_boxes', {})
            corrected_boxes = {}
            corrected_defects = []
            
            # Apply type corrections first
            if type_corrections:
                print("Applying defect type corrections...")
                
                # Find the most confident type correction
                best_correction = None
                for corrected_type, correction_info in type_corrections.items():
                    if not best_correction:
                        best_correction = (corrected_type, correction_info)
                    # Could add logic to pick best correction if multiple
                
                if best_correction:
                    corrected_type, correction_info = best_correction
                    print(f"Correcting defect type to: {corrected_type} - {correction_info['reason']}")
                    
                    # Find the best existing detection to convert
                    best_existing = None
                    best_score = 0
                    
                    for existing_type, boxes in bounding_boxes.items():
                        if boxes:
                            # Score based on confidence and area reasonableness
                            box = boxes[0]
                            confidence = box.get('confidence', 0)
                            area_pct = box.get('area_percentage', 0)
                            
                            # Prefer reasonable sized detections
                            if corrected_type == 'open' and 1 < area_pct < 20:
                                score = confidence + 0.5
                            elif corrected_type == 'scratch' and 0.1 < area_pct < 10:
                                score = confidence + 0.5
                            else:
                                score = confidence
                            
                            if score > best_score:
                                best_score = score
                                best_existing = (existing_type, box)
                    
                    if best_existing:
                        existing_type, existing_box = best_existing
                        print(f"Converting {existing_type} detection to {corrected_type}")
                        
                        # Create corrected box
                        corrected_box = existing_box.copy()
                        corrected_box.update({
                            'openai_type_corrected': True,
                            'original_type': existing_type,
                            'corrected_type': corrected_type,
                            'correction_reason': correction_info['reason']
                        })
                        
                        # Apply bbox correction if available for this type
                        if corrected_type in bbox_corrections:
                            bbox_correction = bbox_corrections[corrected_type]
                            print(f"Also applying bbox correction for {corrected_type}")
                            
                            corrected_box.update({
                                'x': bbox_correction['x'],
                                'y': bbox_correction['y'],
                                'width': bbox_correction['width'],
                                'height': bbox_correction['height'],
                                'center_x': bbox_correction['x'] + bbox_correction['width'] // 2,
                                'center_y': bbox_correction['y'] + bbox_correction['height'] // 2,
                                'area': bbox_correction['width'] * bbox_correction['height'],
                                'openai_bbox_corrected': True,
                                'bbox_correction_reason': bbox_correction['reason']
                            })
                            
                            # Recalculate area percentage
                            if 'predicted_mask' in result:
                                total_pixels = result['predicted_mask'].shape[0] * result['predicted_mask'].shape[1]
                                corrected_box['area_percentage'] = (corrected_box['area'] / total_pixels) * 100
                        
                        corrected_boxes[corrected_type] = [corrected_box]
                        corrected_defects.append(corrected_type)
                    else:
                        print(f"No suitable existing detection found to convert to {corrected_type}")
            
            # If no type corrections applied, apply bbox corrections to existing types
            if not corrected_boxes:
                for defect_type, boxes in bounding_boxes.items():
                    if defect_type in bbox_corrections and boxes:
                        correction = bbox_corrections[defect_type]
                        print(f"Applying bbox correction for {defect_type}: {correction['reason']}")
                        
                        corrected_box = boxes[0].copy()
                        corrected_box.update({
                            'x': correction['x'],
                            'y': correction['y'],
                            'width': correction['width'],
                            'height': correction['height'],
                            'center_x': correction['x'] + correction['width'] // 2,
                            'center_y': correction['y'] + correction['height'] // 2,
                            'area': correction['width'] * correction['height'],
                            'openai_bbox_corrected': True,
                            'correction_reason': correction['reason']
                        })
                        
                        # Recalculate area percentage
                        if 'predicted_mask' in result:
                            total_pixels = result['predicted_mask'].shape[0] * result['predicted_mask'].shape[1]
                            corrected_box['area_percentage'] = (corrected_box['area'] / total_pixels) * 100
                        
                        corrected_boxes[defect_type] = [corrected_box]
                    else:
                        corrected_boxes[defect_type] = boxes
                
                corrected_defects = list(corrected_boxes.keys())
            
            # Update result with corrections
            if corrected_boxes:
                result['bounding_boxes'] = corrected_boxes
                result['detected_defects'] = corrected_defects
                
                if 'defect_analysis' in result:
                    result['defect_analysis']['bounding_boxes'] = corrected_boxes
                    result['defect_analysis']['detected_defects'] = corrected_defects
            
            result['openai_corrections_applied'] = True
            result['corrections_summary'] = {
                'type_corrections_count': len(type_corrections),
                'bbox_corrections_count': len(bbox_corrections),
                'final_defect_types': corrected_defects
            }
            
            return result
            
        except Exception as e:
            print(f"Error applying OpenAI corrections: {e}")
            return result
    
    def _encode_image_to_base64(self, image_path):
        """Encode image to base64 for OpenAI"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_confidence_percentage(self, text):
        """Extract general confidence percentage from OpenAI response"""
        import re
        matches = re.findall(r'(\d+)%', text)
        if matches:
            return max([int(match) for match in matches])
        return 75  # Default confidence
    
    def _extract_bbox_confidence(self, text):
        """Extract bounding box confidence from OpenAI response"""
        import re
        
        # Look for bounding box specific confidence patterns
        bbox_patterns = [
            r'bounding box.*?(\d+)%',
            r'spatial.*?(\d+)%',
            r'location.*?(\d+)%',
            r'bbox.*?(\d+)%',
            r'accuracy.*?(\d+)%',
            r'boxes.*?(\d+)%',
            r'positioning.*?(\d+)%'
        ]
        
        confidences = []
        for pattern in bbox_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            confidences.extend([int(match) for match in matches])
        
        if confidences:
            return max(confidences)
        
        # Fallback to general confidence if no bbox-specific found
        general_matches = re.findall(r'(\d+)%', text)
        if general_matches:
            return max([int(match) for match in general_matches])
        
        return 75  # Default confidence
    
    def _analyze_defect_predictions(self, predicted_mask, confidence_scores):
        """Analyze HRNet predictions to extract defect information"""
        h, w = predicted_mask.shape
        total_pixels = h * w
        
        analysis = {
            'detected_defects': [],
            'class_distribution': {},
            'bounding_boxes': {},
            'defect_statistics': {}
        }
        
        # Analyze each defect class
        for class_id, class_name in SPECIFIC_DEFECT_CLASSES.items():
            class_mask = (predicted_mask == class_id)
            pixel_count = np.sum(class_mask)
            percentage = (pixel_count / total_pixels) * 100
            
            analysis['class_distribution'][class_name] = {
                'pixel_count': int(pixel_count),
                'percentage': percentage,
                'class_id': class_id
            }
            
            # Only process actual defects (not background)
            if class_id > 0 and pixel_count > 0:
                # Apply confidence threshold
                confident_mask = class_mask & (confidence_scores > DEFECT_CONFIDENCE_THRESHOLD)
                confident_pixels = np.sum(confident_mask)
                
                # Detection criteria
                min_pixels = max(MIN_DEFECT_PIXELS, total_pixels * MIN_DEFECT_PERCENTAGE)
                
                if confident_pixels > min_pixels or pixel_count > total_pixels * 0.1:
                    analysis['detected_defects'].append(class_name)
                    
                    # Extract bounding boxes
                    bboxes = self._extract_bounding_boxes(
                        confident_mask if confident_pixels > min_pixels else class_mask
                    )
                    analysis['bounding_boxes'][class_name] = bboxes
                    
                    # Calculate statistics
                    analysis['defect_statistics'][class_name] = {
                        'confident_pixels': int(confident_pixels if confident_pixels > 0 else pixel_count),
                        'confidence_ratio': confident_pixels / pixel_count if pixel_count > 0 else 0,
                        'avg_confidence': float(np.mean(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'max_confidence': float(np.max(confidence_scores[
                            confident_mask if confident_pixels > 0 else class_mask
                        ])),
                        'num_regions': len(bboxes)
                    }
        
        return analysis
    
    def _extract_bounding_boxes(self, mask):
        """Extract bounding boxes from binary mask"""
        try:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bounding_boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= MIN_BBOX_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append({
                        'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                        'area': int(area),
                        'center_x': int(x + w/2), 'center_y': int(y + h/2)
                    })
            
            return bounding_boxes
        except:
            return []
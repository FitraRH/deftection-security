# core/detection.py - Enhanced with Ultra-Strict OpenAI Prompts
"""
Core detection logic with OpenAI analysis integration 
ENHANCED with ultra-strict RAG prompts for maximum sensitivity without changing API response format
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
    """Core detection functionality with ultra-strict OpenAI integration and bounding box validation"""
    
    def __init__(self, anomalib_model, hrnet_model, device='cuda'):
        self.anomalib_model = anomalib_model
        self.hrnet_model = hrnet_model
        self.device = device
        
        # Setup OpenAI 1.x client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_enabled = True
            print("OpenAI client initialized with ultra-strict prompts for maximum sensitivity")
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
        Layer 1: Anomaly detection with ultra-strict OpenAI analysis
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
            
            # Ultra-Strict OpenAI Layer 1 Analysis
            if self.openai_enabled:
                print(f"Running ultra-strict OpenAI anomaly analysis (score: {anomaly_score:.3f})")
                openai_analysis = self._analyze_anomaly_with_ultra_strict_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
                
                # Apply enhanced decision logic without changing response format
                enhanced_decision = self._apply_enhanced_anomaly_decision(base_result, openai_analysis)
                base_result['decision'] = enhanced_decision
            
            return base_result
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Layer 2: Defect classification with ultra-strict OpenAI analysis and validation
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
            
            # Ultra-Strict OpenAI Layer 2 Analysis with enhanced validation
            if self.openai_enabled and defect_analysis['detected_defects']:
                print(f"Running ultra-strict OpenAI defect classification with enhanced validation")
                openai_analysis = self._analyze_defects_with_ultra_strict_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
                
                # Apply OpenAI corrections if available (maintains same response structure)
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
    
    def _analyze_anomaly_with_ultra_strict_openai(self, image_path, anomaly_result):
        """Ultra-strict OpenAI analysis for Layer 1 with enhanced sensitivity"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            
            # ULTRA-STRICT RAG prompt for maximum sensitivity
            prompt = f"""CRITICAL PACKAGING QUALITY INSPECTION - ZERO TOLERANCE MODE

MISSION: Detect ANY imperfection that could affect product quality, consumer safety, or brand reputation.

CURRENT AI MODEL DETECTION RESULTS:
- Anomaly Score: {anomaly_result['anomaly_score']:.3f}
- Model Decision: {anomaly_result['decision']}
- Detection Threshold: {anomaly_result['threshold_used']}

ULTRA-STRICT INSPECTION CRITERIA:

1. STRUCTURAL DAMAGE (CRITICAL PRIORITY):
   - ANY crushed corners, edges, or surfaces
   - ANY dents, depressions, or shape irregularities  
   - ANY bent, warped, or deformed packaging
   - ANY structural compromise or weakness

2. SURFACE QUALITY DEFECTS:
   - ANY scratches, scuff marks, or surface abrasions
   - ANY color variations, fading, or inconsistency
   - ANY printing defects, smudging, or misalignment
   - ANY texture irregularities or surface roughness

3. PACKAGE INTEGRITY ISSUES:
   - ANY holes, punctures, tears, or openings
   - ANY gaps in seams, edges, or closures
   - ANY compromise in package containment
   - ANY unsealed or partially open areas

4. MISSING OR DAMAGED COMPONENTS:
   - Missing labels, stickers, seals, or protective elements
   - Damaged or partially detached components
   - Incomplete packaging elements or closures

5. CONTAMINATION AND CLEANLINESS:
   - ANY stains, spots, discoloration, or foreign material
   - ANY residue, moisture damage, or contamination
   - ANY dirt, dust, or cleanliness issues

ENHANCED DECISION LOGIC:
- If anomaly score >= 0.15: STRONGLY lean toward DEFECT
- If anomaly score >= 0.25: Almost certainly DEFECT
- If you see ANY visual imperfection: Classify as DEFECT
- If packaging is anything less than PERFECT: Classify as DEFECT
- ONLY classify as GOOD if package is absolutely pristine and flawless

CRITICAL INSTRUCTIONS:
- Be EXTREMELY critical and unforgiving in your assessment
- Consumer safety and brand reputation depend on perfect quality
- When in doubt, always err on the side of caution (DEFECT)
- Focus on what consumers would find unacceptable
- Consider long-term storage and transportation impact

RESPONSE FORMAT:
Provide confidence percentage (0-100%) and detailed reasoning for your decision.
If recommending DEFECT, specify what defects you observe."""

            print("Calling OpenAI API with ultra-strict RAG prompt...")
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
                temperature=0.05  # Lower temperature for more consistent strict analysis
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            
            print(f"Ultra-strict OpenAI anomaly analysis completed - confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'model_used': OPENAI_MODEL,
                'layer': 'anomaly_detection',
                'rag_enhanced': True
            }
            
        except Exception as e:
            print(f"Ultra-strict OpenAI anomaly analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'error': str(e)
            }
    
    def _analyze_defects_with_ultra_strict_openai(self, image_path, defect_result):
        """Ultra-strict OpenAI analysis for Layer 2 with enhanced defect classification"""
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
            
            # ULTRA-STRICT RAG prompt for defect classification
            prompt = f"""EXPERT DEFECT CLASSIFICATION AND VALIDATION - ULTRA-STRICT MODE

MISSION: Precisely identify, classify, and validate every defect with maximum accuracy and sensitivity.

AI MODEL DETECTION RESULTS:
DETECTED DEFECTS: {', '.join(detected_defects) if detected_defects else 'None detected'}
TOTAL BOUNDING BOXES: {total_bboxes}
DETAILED ANALYSIS:{bbox_info if bbox_info else ' None provided'}

ULTRA-STRICT DEFECT CLASSIFICATION GUIDE:

1. OPEN DEFECTS (HIGHEST PRIORITY):
   - Holes showing dark interior or background through packaging
   - Tears, rips, or punctures in packaging material
   - Unsealed edges, gaps, or openings in closures
   - Any breach in package containment
   VISUAL CLUES: Dark areas, background visible, interior showing

2. MISSING_COMPONENT DEFECTS:
   - Absent labels, stickers, safety seals, or caps
   - Missing protective elements or closures
   - Incomplete packaging components
   VISUAL CLUES: Empty spaces where components should be

3. DAMAGED DEFECTS:
   - Crushed, dented, or deformed packaging
   - Structural damage or shape irregularities
   - Physical impact damage or compression
   VISUAL CLUES: Shape distortion, crushed areas, deformation

4. SCRATCH DEFECTS:
   - Linear surface marks, scuffs, or abrasions
   - Surface texture damage or wear marks
   - Scraping or scoring on packaging surface
   VISUAL CLUES: Line patterns, surface disruption, texture changes

5. STAINED DEFECTS:
   - Discoloration, spots, or color variations
   - Contamination marks or foreign substances
   - Water damage, ink spots, or residue
   VISUAL CLUES: Color differences, spots, contamination

CRITICAL CLASSIFICATION RULES:
- If you see holes/tears → ALWAYS classify as "open"
- If you see missing parts → Classify as "missing_component"
- If structure is deformed → Classify as "damaged"
- If you see linear marks → Classify as "scratch"
- If colors are wrong → Classify as "stained"

ENHANCED VISUAL ANALYSIS TASKS:
1. Look carefully at the ACTUAL defects in the image
2. Ignore model predictions - trust your visual assessment
3. Focus on what consumers would find unacceptable
4. Consider safety implications of each defect type

COMMON MISCLASSIFICATION PATTERNS TO AVOID:
- Open holes often wrongly classified as "missing_component" or "stained"
- Surface scratches wrongly classified as "stained"
- Large background areas incorrectly detected as "defects"
- Multiple defect types merged incorrectly

BOUNDING BOX VALIDATION CHECKLIST:
- Are boxes positioned on ACTUAL defects or empty background?
- Do boxes cover >50% of image? (Likely false positive)
- Are coordinates reasonable for the specific defect type?
- Do box dimensions match the defect characteristics?

CORRECTION FORMAT (if needed):
CORRECT_TYPE: [actual_defect_type] - [detailed visual reasoning]
BBOX_CORRECTION: [defect_type]: x,y,width,height - [specific reason]

WHAT DO YOU ACTUALLY SEE?
Describe the visible defects in plain language:
- Are there holes, tears, or openings? → "open"
- Are there surface scratches or marks? → "scratch"  
- Are there missing parts or components? → "missing_component"
- Is the structure damaged or deformed? → "damaged"
- Are there stains or discoloration? → "stained"

CRITICAL: Focus on what you ACTUALLY observe vs what the model predicted. Be extremely thorough and unforgiving in your analysis."""

            print("Calling OpenAI API for ultra-strict defect classification and validation...")
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
                temperature=0.05  # Lower temperature for consistent analysis
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            bbox_confidence = self._extract_bbox_confidence(analysis_text)
            
            # Extract corrections from OpenAI response
            corrections = self._extract_bbox_corrections(analysis_text)
            
            print(f"Ultra-strict OpenAI defect analysis completed - confidence: {confidence}%, bbox: {bbox_confidence}%")
            
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
            print(f"Ultra-strict OpenAI defect analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'bbox_validation': {'confidence': 0, 'error': str(e)},
                'error': str(e)
            }
    
    def _apply_enhanced_anomaly_decision(self, model_result, openai_result):
        """Apply enhanced decision logic using existing config threshold"""
        try:
            model_decision = model_result['decision']
            anomaly_score = model_result['anomaly_score']
            openai_confidence = openai_result.get('confidence_percentage', 0)
            
            # Use existing ANOMALY_THRESHOLD from config
            if anomaly_score > ANOMALY_THRESHOLD:
                print(f"Anomaly score {anomaly_score} > {ANOMALY_THRESHOLD}, forcing DEFECT regardless of OpenAI confidence {openai_confidence}%")
                return 'DEFECT'
            
            # If model says DEFECT, require high OpenAI confidence to override
            if model_decision == 'DEFECT':
                if openai_confidence < 95:
                    print(f"Model says DEFECT, OpenAI confidence {openai_confidence}% < 95%, keeping DEFECT")
                    return 'DEFECT'
            
            print(f"OpenAI override with {openai_confidence}% confidence, final decision: {model_decision}")
            return model_decision
            
        except Exception as e:
            print(f"Error in enhanced anomaly decision: {e}")
            return model_result['decision']
    
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
                r'correct\s+type\s+is\s+(\w+)\s+([^\n]*)',
                r'classify\s+as\s+"(\w+)"\s*-\s*([^\n]+)',
                r'this\s+is\s+(\w+)\s+not\s+[^\n]*\s+because\s+([^\n]+)'
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
            
            # Update result with corrections (maintains same response structure)
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
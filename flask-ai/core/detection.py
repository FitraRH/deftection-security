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
    """Core detection functionality with product-aware OpenAI integration"""
    
    def __init__(self, anomalib_model, hrnet_model, device='cuda'):
        self.anomalib_model = anomalib_model
        self.hrnet_model = hrnet_model
        self.device = device
        
        # Setup OpenAI 1.x client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.openai_enabled = True
            print("OpenAI client initialized with product-aware prompts")
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
        Layer 1: Anomaly detection with product-aware OpenAI analysis
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
            
            # Product-Aware OpenAI Layer 1 Analysis
            if self.openai_enabled:
                print(f"Running product-aware OpenAI anomaly analysis (score: {anomaly_score:.3f})")
                openai_analysis = self._analyze_anomaly_with_product_aware_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
                
                # Apply enhanced decision logic with product context
                enhanced_decision = self._apply_product_aware_anomaly_decision(base_result, openai_analysis)
                base_result['decision'] = enhanced_decision
            
            return base_result
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None
    
    def classify_defects(self, image_path, region_mask=None):
        """
        Layer 2: Defect classification with product-aware OpenAI analysis
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
            
            # Use enhanced detection
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
            
            # Product-Aware OpenAI Layer 2 Analysis
            if self.openai_enabled and defect_analysis['detected_defects']:
                print(f"Running product-aware OpenAI defect classification")
                openai_analysis = self._analyze_defects_with_product_aware_openai(image_path, base_result)
                base_result['openai_analysis'] = openai_analysis
                
                # Apply OpenAI corrections with product context
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
    
    # Modified RAG Prompts untuk Anomalib + HRNet Pipeline

    def _analyze_anomaly_with_product_aware_openai(self, image_path, anomaly_result):
        """Enhanced RAG prompt untuk Anomalib - fokus ke anomaly score validation"""
        try:
            if not self.openai_client:
                return {
                    'analysis': 'OpenAI client not initialized',
                    'confidence_percentage': 0,
                    'error': 'No OpenAI client'
                }
            
            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            
            # ENHANCED ANOMALIB RAG PROMPT
            prompt = f"""PRODUCT-AWARE PACKAGING ANOMALY VALIDATION

    MISSION: Validate anomaly detection results ONLY for manufactured packaging/products. Filter out non-packaging false positives.

    ANOMALIB MODEL RESULTS:
    - Anomaly Score: {anomaly_result['anomaly_score']:.3f}
    - Model Decision: {anomaly_result['decision']}
    - Detection Threshold: {anomaly_result['threshold_used']}

    
    STEP 1: PRODUCT CONTEXT VERIFICATION
    Determine if this image shows a MANUFACTURED PRODUCT or PACKAGE:

    VALID PACKAGING ITEMS (eligible for anomaly analysis):
     - Consumer product packaging (boxes, bottles, cans, pouches, wrappers)
     - Electronics packaging (phone boxes, device cases, charger boxes)
     - Food packaging (snack bags, cereal boxes, beverage containers)
     - Pharmaceutical packaging (medicine bottles, blister packs, pill boxes)
     - Cosmetic packaging (bottles, tubes, compact cases, containers)
     - Industrial products with visible packaging or protective casing
     - Any manufactured item with identifiable packaging/wrapping

    INVALID ITEMS (always return GOOD - not packaging anomalies):
     - Building surfaces (walls, floors, ceilings, doors, windows)
     - Furniture or home decor (tables, chairs, sofas, decorations)
     - Natural objects (rocks, trees, sky, landscapes, water)
     - People, animals, or body parts
     - Vehicles (cars, motorcycles, bicycles, transportation)
     - Abstract art, drawings, animations, or digital graphics
     - Screens, monitors, TVs, or digital displays
     - Tools, equipment, or machinery (unless clearly packaged)
     - Raw materials, fabric, or unprocessed items
     - Empty backgrounds, plain surfaces, or textured walls

   
    STEP 2: ANOMALY SCORE VALIDATION (Only if Step 1 = VALID PACKAGING)

    If this IS a manufactured product/package, validate the anomaly detection:

    ANOMALY INDICATORS TO VERIFY:
    1. STRUCTURAL ANOMALIES:
    - Unusual shape deformation or crushing
    - Unexpected cracks, tears, or holes
    - Bent or damaged structural elements

    2. SURFACE ANOMALIES:
    - Unusual scratches, scuffs, or surface marks
    - Inconsistent print quality or alignment
    - Unexpected color variations or discoloration

    3. COMPLETENESS ANOMALIES:
    - Missing expected components (labels, caps, seals)
    - Incomplete or damaged closures
    - Absent elements that should be present

    4. CONTAMINATION ANOMALIES:
    - Unexpected stains, spots, or foreign substances
    - Dirt, debris, or contamination
    - Unusual discoloration patterns

   
    STEP 3: ANOMALY SCORE INTERPRETATION

    Current Score: {anomaly_result['anomaly_score']:.3f} (Threshold: {anomaly_result['threshold_used']})

    SCORE VALIDATION GUIDELINES:
    - Score > 0.8: Very likely anomaly (if valid packaging)
    - Score 0.5-0.8: Moderate anomaly (verify carefully)
    - Score 0.3-0.5: Low anomaly (might be normal variation)
    - Score < 0.3: Likely normal (unless obvious defects visible)

    DECISION LOGIC:
    - If NON-PACKAGING item → ALWAYS return GOOD (ignore anomaly score)
    - If VALID PACKAGING → Validate anomaly score against visible issues
    - Consider model score as supporting evidence for VALID products only
    - Be strict about product context - when uncertain, default to GOOD

    ========================================================
    CRITICAL VALIDATION RULES:
    1. This system is trained SPECIFICALLY for PACKAGING ANOMALY DETECTION
    2. Non-packaging items should NEVER be flagged as anomalous
    3. Empty rooms, walls, random objects are NOT packaging anomalies
    4. Only analyze items that could realistically have packaging defects
    5. Anomaly scores are only meaningful for manufactured products

    ========================================================
    RESPONSE FORMAT:
    Start with: "PRODUCT TYPE: [VALID_PACKAGING / INVALID_NON_PACKAGING]"
    Then provide:
    - Classification: [GOOD / DEFECT]
    - Confidence percentage (0-100%)
    - Detailed reasoning for decision
    - If INVALID_NON_PACKAGING: "All anomaly detections rejected - not applicable for packaging inspection"
    - If VALID_PACKAGING: Explain whether anomaly score aligns with visible issues"""

            print("Calling OpenAI API with enhanced Anomalib RAG prompt...")
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
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            product_type = self._extract_product_type(analysis_text)
            classification = self._extract_classification(analysis_text)
            
            print(f"Enhanced Anomalib analysis completed - Type: {product_type}, Classification: {classification}, Confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'product_type': product_type,
                'classification': classification,
                'model_used': OPENAI_MODEL,
                'layer': 'anomaly_detection',
                'product_aware': True,
                'anomaly_score_validated': True
            }
            
        except Exception as e:
            print(f"Enhanced Anomalib analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'product_type': 'unknown',
                'classification': 'unknown',
                'error': str(e)
            }

    def _analyze_defects_with_product_aware_openai(self, image_path, defect_result):
        """Enhanced RAG prompt untuk HRNet - fokus ke mask/segmentation validation"""
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
            
            # Create detailed mask information from segmentation results
            mask_info = ""
            total_masks = 0
            for defect_type, boxes in bounding_boxes.items():
                if defect_type == 'background':
                    continue
                mask_info += f"\n{defect_type.upper()}: {len(boxes)} segmented regions"
                total_masks += len(boxes)
                for i, bbox in enumerate(boxes):
                    area_pct = bbox.get('area_percentage', 0)
                    conf = bbox.get('confidence', 0)
                    compactness = bbox.get('compactness', 0)
                    shape_type = bbox.get('shape_type', 'unknown')
                    mask_info += f"\n  Region {i+1}: Area({area_pct:.1f}%) Conf({conf:.3f}) Shape({shape_type}) Compactness({compactness:.2f})"
            
            # ENHANCED HRNET RAG PROMPT - FOKUS KE SEGMENTATION MASKS
            prompt = f"""PRODUCT-AWARE DEFECT TYPE CLASSIFICATION & MASK VALIDATION

    MISSION: Validate HRNet segmentation masks and classify defect types ONLY for legitimate packaging products.

    HRNET SEGMENTATION RESULTS:
    DETECTED DEFECT TYPES: {', '.join(detected_defects) if detected_defects else 'None detected'}
    TOTAL SEGMENTED REGIONS: {total_masks}
    MASK ANALYSIS:{mask_info if mask_info else ' None provided'}

    
    STEP 1: PRODUCT CONTEXT VERIFICATION
    Verify this image shows a MANUFACTURED PRODUCT/PACKAGE suitable for defect segmentation:

    LEGITIMATE PACKAGING PRODUCTS (continue mask validation):
     - Consumer product packaging (boxes, containers, wrappers, pouches)
     - Electronics and device packaging (cases, boxes)
     - Food and beverage packaging (bags, bottles, cartons)
     - Medical/pharmaceutical packaging (bottles, blister packs)
     - Cosmetic and personal care packaging (tubes, bottles)
     - Industrial products with visible packaging/casing

    NON-PACKAGING ITEMS (reject all masks):
     - Building surfaces (walls, floors, ceilings)
     - Furniture or home items (tables, chairs, decor)
     - Natural scenes or landscapes
     - People, animals, or organic subjects
     - Vehicles or transportation objects
     - Art, animations, or digital graphics
     - Tools or equipment (unless packaged)
     - Raw materials or unfinished items

   
    STEP 2: SEGMENTATION MASK VALIDATION (Only for legitimate products)

    If this IS a legitimate product, validate each segmentation mask:

    1. OPEN DEFECTS (holes, tears, punctures):
    - Mask should highlight actual holes/tears/breaches in packaging
    - Shape: Irregular, often on edges or structural weak points
    - Pattern: Clear boundaries between intact and damaged areas
    - NOT: Normal design features, intended openings, or transparent windows

    2. SCRATCH DEFECTS (surface abrasions):
    - Mask should be thin/linear, following surface damage patterns
    - Shape: Linear, elongated, or streak-like segmentation
    - Pattern: Aligned with direction of physical contact/abrasion
    - NOT: Design elements, text lines, barcodes, or intentional patterns

    3. STAINED DEFECTS (discoloration, contamination):
    - Mask should cover irregular patches of unwanted discoloration
    - Shape: Blotchy, irregular, or spread-pattern segmentation
    - Pattern: Non-uniform edges, varying intensity
    - NOT: Normal coloring, shadows, logos, or intended design colors

    4. DAMAGED DEFECTS (structural deformation):
    - Mask should cover bent, crushed, or structurally deformed areas
    - Shape: Large irregular regions, often covering significant area
    - Pattern: Follows structural damage contours
    - NOT: Normal product shape variations or intended contours

    5. MISSING_COMPONENT DEFECTS (absent parts):
    - Mask should indicate empty regions where components should exist
    - Shape: Well-defined areas matching expected component shapes
    - Pattern: Clear absence of expected elements (labels, caps, seals)
    - NOT: Design variations or alternative product configurations

  
    STEP 3: MASK QUALITY & SPATIAL VALIDATION

    For each segmentation mask, verify:

    SPATIAL VALIDATION:
    - Is the mask positioned ON the actual product (not background)?
    - Does the mask shape align with expected defect morphology?
    - Is the mask size proportional to the defect type?
    - Are mask boundaries sharp and well-defined (not noisy)?

    CONTEXTUAL VALIDATION:
    - Does the segmented area represent a REAL defect vs normal variation?
    - Is the mask covering genuine packaging issues vs design elements?
    - Are multiple small masks better represented as one larger defect?
    - Do masks overlap with text, logos, or intentional design features?

   
    STEP 4: DEFECT TYPE CONFIDENCE ASSESSMENT

    For each validated mask:
    - Assess how well the mask shape matches the classified defect type
    - Consider mask compactness and spatial distribution
    - Evaluate segmentation clarity and boundary definition
    - Rate confidence in defect type classification (0-100%)

   
    CRITICAL VALIDATION RULES:
    1. This system is SPECIFICALLY for PACKAGING DEFECT SEGMENTATION
    2. Non-packaging items should have ALL masks rejected
    3. For packaging items, only validate masks covering ACTUAL defects
    4. Background areas, normal features, and design elements are NOT defects
    5. Prioritize mask quality and spatial accuracy over quantity
    6. When uncertain about product relevance, reject the segmentation

    
    RESPONSE FORMAT:
    Start with: "PRODUCT VALIDATION: [LEGITIMATE_PACKAGING / NON_PACKAGING_ITEM]"

    If NON_PACKAGING_ITEM: 
    "All segmentation masks rejected - not applicable for packaging inspection"

    If LEGITIMATE_PACKAGING:
    For each segmentation mask, provide:
    - Defect Type: [Open/Scratch/Stained/Damaged/Missing_Component]
    - Mask Quality: [Excellent/Good/Fair/Poor]
    - Spatial Accuracy: [High/Medium/Low]
    - Confidence: [0-100%]
    - Validation Decision: [ACCEPT/REJECT]
    - Reasoning: Detailed explanation for acceptance or rejection

    Include overall confidence percentage and comprehensive validation summary."""

            print("Calling OpenAI API with enhanced HRNet segmentation RAG prompt...")
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
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            confidence = self._extract_confidence_percentage(analysis_text)
            mask_confidence = self._extract_mask_confidence(analysis_text)
            product_validation = self._extract_product_validation(analysis_text)
            
            # Extract mask-specific corrections from OpenAI response
            corrections = self._extract_mask_corrections(analysis_text)
            
            print(f"Enhanced HRNet segmentation analysis completed - Product: {product_validation}, Confidence: {confidence}%")
            
            return {
                'analysis': analysis_text,
                'confidence_percentage': confidence,
                'product_validation': product_validation,
                'mask_validation': {
                    'confidence': mask_confidence,
                    'validated_regions': len(bounding_boxes),
                    'spatial_accuracy': 'high' if mask_confidence > 80 else 'medium' if mask_confidence > 60 else 'low',
                    'segmentation_quality': 'excellent' if mask_confidence > 90 else 'good' if mask_confidence > 70 else 'fair'
                },
                'mask_corrections': corrections.get('mask_corrections', {}),
                'type_corrections': corrections.get('type_corrections', {}),
                'model_used': OPENAI_MODEL,
                'layer': 'defect_classification',
                'defects_analyzed': detected_defects,
                'segmentation_masks_analyzed': {k: len(v) for k, v in bounding_boxes.items()},
                'product_aware': True,
                'segmentation_validation': True,
                'mask_spatial_validation': True,
                'context_filtering': True
            }
            
        except Exception as e:
            print(f"Enhanced HRNet segmentation analysis error: {e}")
            return {
                'analysis': f'OpenAI analysis failed: {str(e)}',
                'confidence_percentage': 0,
                'product_validation': 'unknown',
                'mask_validation': {'confidence': 0, 'error': str(e)},
                'error': str(e)
            }

    # Helper methods untuk extract information dari OpenAI responses
    def _extract_classification(self, analysis_text):
        """Extract classification (GOOD/DEFECT) from OpenAI response"""
        try:
            import re
            
            # Look for classification indicators
            if re.search(r'CLASSIFICATION:\s*GOOD', analysis_text, re.IGNORECASE):
                return 'GOOD'
            elif re.search(r'CLASSIFICATION:\s*DEFECT', analysis_text, re.IGNORECASE):
                return 'DEFECT'
            elif re.search(r'return\s+GOOD|conclude\s+GOOD', analysis_text, re.IGNORECASE):
                return 'GOOD'
            elif re.search(r'return\s+DEFECT|conclude\s+DEFECT', analysis_text, re.IGNORECASE):
                return 'DEFECT'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error extracting classification: {e}")
            return 'unknown'

    def _extract_mask_confidence(self, analysis_text):
        """Extract mask-specific confidence from OpenAI response"""
        try:
            import re
            
            # Look for mask-specific confidence patterns
            mask_patterns = [
                r'mask.*?confidence.*?(\d+)%',
                r'segmentation.*?confidence.*?(\d+)%',
                r'spatial.*?accuracy.*?(\d+)%',
                r'mask.*?quality.*?(\d+)%'
            ]
            
            confidences = []
            for pattern in mask_patterns:
                matches = re.findall(pattern, analysis_text, re.IGNORECASE)
                confidences.extend([int(match) for match in matches])
            
            if confidences:
                return max(confidences)
            
            # Fallback to general confidence
            general_matches = re.findall(r'(\d+)%', analysis_text)
            if general_matches:
                return max([int(match) for match in general_matches])
            
            return 75  # Default confidence
            
        except Exception as e:
            print(f"Error extracting mask confidence: {e}")
            return 75

    def _extract_mask_corrections(self, analysis_text):
        """Extract mask corrections from OpenAI analysis"""
        corrections = {}
        type_corrections = {}
        
        try:
            import re
            
            # Extract mask-specific corrections
            mask_correction_patterns = [
                r'MASK_CORRECTION:\s*(\w+):\s*([^\n]+)',
                r'REJECT\s+(\w+)\s+mask:\s*([^\n]+)',
                r'ACCEPT\s+(\w+)\s+mask:\s*([^\n]+)',
                r'(\w+)\s+mask\s+should\s+be\s+([^\n]+)'
            ]
            
            for pattern in mask_correction_patterns:
                matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    defect_type = match.group(1).lower()
                    correction_info = match.group(2)
                    
                    corrections[defect_type] = {
                        'correction_info': correction_info,
                        'source': 'openai_mask_validation'
                    }
            
            # Extract type corrections for segmentation
            type_patterns = [
                r'should\s+be\s+classified\s+as\s+(\w+)\s*-\s*([^\n]+)',
                r'actually\s+(\w+)\s+defect\s+([^\n]*)',
                r'reclassify\s+as\s+"(\w+)"\s*-\s*([^\n]+)'
            ]
            
            for pattern in type_patterns:
                matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    defect_type = match.group(1).lower()
                    reason = match.group(2) if len(match.groups()) > 1 else "OpenAI mask correction"
                    
                    # Validate defect type
                    valid_types = ['open', 'scratch', 'missing_component', 'damaged', 'stained']
                    if defect_type in valid_types:
                        type_corrections[defect_type] = {
                            'corrected_type': defect_type,
                            'reason': reason,
                            'source': 'openai_mask_type_correction'
                        }
            
            return {
                'mask_corrections': corrections,
                'type_corrections': type_corrections
            }
            
        except Exception as e:
            print(f"Error extracting mask corrections: {e}")
            return {'mask_corrections': {}, 'type_corrections': {}}
    
    def _apply_product_aware_anomaly_decision(self, model_result, openai_result):
        """Apply product-aware decision logic - reject non-packaging items"""
        try:
            model_decision = model_result['decision']
            anomaly_score = model_result['anomaly_score']
            openai_confidence = openai_result.get('confidence_percentage', 0)
            product_type = openai_result.get('product_type', 'unknown')
            
            # If OpenAI determines this is not a packaging item, force GOOD
            if product_type == 'INVALID_NON_PACKAGING':
                print(f"Product-aware: Non-packaging item detected, forcing GOOD decision")
                return 'GOOD'
            
            # If OpenAI is uncertain about product type but confidence is low, be conservative
            if product_type == 'unknown' and openai_confidence < 50:
                print(f"Product-aware: Uncertain product type with low confidence, being conservative")
                return 'GOOD'
            
            # For valid packaging items, use normal logic
            if anomaly_score > ANOMALY_THRESHOLD:
                print(f"Product-aware: Valid packaging with anomaly score {anomaly_score} > {ANOMALY_THRESHOLD}, decision: DEFECT")
                return 'DEFECT'
            
            # Conservative approach for packaging items
            if model_decision == 'DEFECT' and product_type == 'VALID_PACKAGING':
                if openai_confidence < 80:  # Higher threshold for packaging validation
                    print(f"Product-aware: Packaging item but OpenAI confidence {openai_confidence}% < 80%, keeping DEFECT")
                    return 'DEFECT'
            
            print(f"Product-aware decision: {model_decision} (Product: {product_type}, Confidence: {openai_confidence}%)")
            return model_decision
            
        except Exception as e:
            print(f"Error in product-aware anomaly decision: {e}")
            return model_result['decision']
    
    def _extract_product_type(self, analysis_text):
        """Extract product type classification from OpenAI response"""
        try:
            import re
            
            # Look for product type indicators
            if re.search(r'PRODUCT TYPE:\s*VALID_PACKAGING', analysis_text, re.IGNORECASE):
                return 'VALID_PACKAGING'
            elif re.search(r'PRODUCT TYPE:\s*INVALID_NON_PACKAGING', analysis_text, re.IGNORECASE):
                return 'INVALID_NON_PACKAGING'
            elif re.search(r'not.*packaging|non.packaging|invalid.*item', analysis_text, re.IGNORECASE):
                return 'INVALID_NON_PACKAGING'
            elif re.search(r'packaging|product|manufactured|container|bottle|box', analysis_text, re.IGNORECASE):
                return 'VALID_PACKAGING'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error extracting product type: {e}")
            return 'unknown'
    
    def _extract_product_validation(self, analysis_text):
        """Extract product validation from OpenAI response"""
        try:
            import re
            
            # Look for product validation indicators
            if re.search(r'PRODUCT VALIDATION:\s*LEGITIMATE_PACKAGING', analysis_text, re.IGNORECASE):
                return 'LEGITIMATE_PACKAGING'
            elif re.search(r'PRODUCT VALIDATION:\s*NON_PACKAGING_ITEM', analysis_text, re.IGNORECASE):
                return 'NON_PACKAGING_ITEM'
            elif re.search(r'not applicable.*packaging|non.packaging|invalid.*product', analysis_text, re.IGNORECASE):
                return 'NON_PACKAGING_ITEM'
            elif re.search(r'legitimate.*packaging|valid.*product|packaging.*item', analysis_text, re.IGNORECASE):
                return 'LEGITIMATE_PACKAGING'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error extracting product validation: {e}")
            return 'unknown'
    
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
                
                if best_correction:
                    corrected_type, correction_info = best_correction
                    print(f"Correcting defect type to: {corrected_type} - {correction_info['reason']}")
                    
                    # Find the best existing detection to convert
                    best_existing = None
                    best_score = 0
                    
                    for existing_type, boxes in bounding_boxes.items():
                        if boxes:
                            box = boxes[0]
                            confidence = box.get('confidence', 0)
                            area_pct = box.get('area_percentage', 0)
                            
                            # Natural scoring without type bias
                            if 1 < area_pct < 25:
                                score = confidence + 0.3
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
            
            # Update result with corrections - maintains same response structure
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
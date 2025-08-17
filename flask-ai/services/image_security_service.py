# services/image_security_service.py - Enhanced Version with YARA Rules
"""
Enhanced Image Security Service - WITH YARA Rules Support
Includes fallback logic when YARA is not available
"""

import os
import hashlib
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# YARA support with fallback
try:
    import yara
    YARA_AVAILABLE = True
    print("YARA-Python is available")
except ImportError:
    YARA_AVAILABLE = False
    print("Warning: YARA-Python not available. Install with: pip install yara-python")

# PIL support with fallback
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available for EXIF analysis")

# Import configuration - with fallback
try:
    from config import (
        MALWARE_HASH_FILE,
        ALLOWED_EXTENSIONS,
        MAX_FILE_SIZE,
        MIN_FILE_SIZE,
        RISK_LEVELS,
        HASH_ALGORITHMS,
        YARA_RULES_DIR
    )
    CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: Could not import security config, using fallback values")
    CONFIG_AVAILABLE = False
    
    # Fallback configuration
    MALWARE_HASH_FILE = "security_data/malware_hashes.txt"
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 100
    RISK_LEVELS = {'CLEAN': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    HASH_ALGORITHMS = ['md5', 'sha1', 'sha256']
    YARA_RULES_DIR = "security_data/yara_rules"

class ImageSecurityService:
    """Enhanced Image Security Service - WITH YARA Rules Support + Fallback"""
    
    def __init__(self):
        """Initialize service with YARA support and fallback capability"""
        self.MALWARE_HASH_FILE = str(MALWARE_HASH_FILE)
        self.YARA_RULES_DIR = str(YARA_RULES_DIR)
        self.ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.RISK_LEVELS = RISK_LEVELS
        
        # Initialize components
        self.malware_hashes = self._load_malware_hashes()
        self.yara_rules = self._load_yara_rules() if YARA_AVAILABLE else None
        
        print(f"Enhanced ImageSecurityService initialized")
        print(f"  Malware hashes loaded: {len(self.malware_hashes)}")
        print(f"  YARA available: {YARA_AVAILABLE}")
        print(f"  YARA rules loaded: {'Yes' if self.yara_rules else 'No'}")
        print(f"  PIL available: {PIL_AVAILABLE}")
        print(f"  Config available: {CONFIG_AVAILABLE}")
    
    def _load_malware_hashes(self) -> set:
        """Load known malware SHA256 hashes"""
        hashes = set()
        
        if os.path.exists(self.MALWARE_HASH_FILE):
            try:
                with open(self.MALWARE_HASH_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) == 64:
                            hashes.add(line.lower())
                print(f"Loaded {len(hashes)} malware hashes")
            except Exception as e:
                print(f"Error loading malware hashes: {e}")
        else:
            print(f"Malware hash file not found (optional): {self.MALWARE_HASH_FILE}")
        
        return hashes
    
    def _load_yara_rules(self) -> Optional[yara.Rules]:
        """Load and compile YARA rules"""
        if not YARA_AVAILABLE:
            return None
        
        try:
            # Create YARA rules directory if it doesn't exist
            os.makedirs(self.YARA_RULES_DIR, exist_ok=True)
            
            # Create sample YARA rules if directory is empty
            self._create_sample_yara_rules()
            
            # Find all .yar and .yara files
            rule_files = []
            for ext in ['*.yar', '*.yara']:
                rule_files.extend(Path(self.YARA_RULES_DIR).glob(ext))
            
            if not rule_files:
                print(f"No YARA rule files found in {self.YARA_RULES_DIR}")
                return None
            
            # Compile rules
            rule_dict = {}
            for rule_file in rule_files:
                try:
                    rule_name = rule_file.stem
                    rule_dict[rule_name] = str(rule_file)
                    print(f"Found YARA rule: {rule_name}")
                except Exception as e:
                    print(f"Error adding rule {rule_file}: {e}")
            
            if rule_dict:
                compiled_rules = yara.compile(filepaths=rule_dict)
                print(f"Successfully compiled {len(rule_dict)} YARA rule files")
                return compiled_rules
            else:
                print("No valid YARA rules to compile")
                return None
                
        except Exception as e:
            print(f"Error loading YARA rules: {e}")
            return None
    
    def _create_sample_yara_rules(self):
        """Create sample YARA rules for demonstration"""
        if not os.path.exists(self.YARA_RULES_DIR):
            os.makedirs(self.YARA_RULES_DIR, exist_ok=True)
        
        # Sample malware detection rules
        sample_rules = {
            "malware_patterns.yar": '''
/*
    Basic Malware Detection Rules
    These are sample rules for demonstration
*/

rule Suspicious_Embedded_PE
{
    meta:
        description = "Detects embedded PE files in images"
        severity = "high"
        author = "Security Scanner"
        
    strings:
        $pe_header = { 4D 5A }  // MZ header
        $pe_signature = "This program cannot be run in DOS mode"
        
    condition:
        $pe_header and $pe_signature
}

rule Suspicious_Script_Content
{
    meta:
        description = "Detects suspicious script content in image metadata"
        severity = "medium"
        
    strings:
        $script1 = "<script" nocase
        $script2 = "eval(" nocase
        $script3 = "document.write" nocase
        $script4 = "javascript:" nocase
        
    condition:
        any of them
}

rule Web_Shell_Patterns
{
    meta:
        description = "Detects web shell patterns"
        severity = "critical"
        
    strings:
        $shell1 = "system($_" nocase
        $shell2 = "exec($_" nocase
        $shell3 = "shell_exec(" nocase
        $shell4 = "passthru(" nocase
        $shell5 = "eval(base64_decode(" nocase
        
    condition:
        any of them
}
            ''',
            
            "steganography_detection.yar": '''
/*
    Steganography Detection Rules
*/

rule Steganography_Tools
{
    meta:
        description = "Detects steganography tool signatures"
        severity = "medium"
        
    strings:
        $steg1 = "steghide" nocase
        $steg2 = "outguess" nocase
        $steg3 = "jsteg" nocase
        $steg4 = "f5stego" nocase
        
    condition:
        any of them
}

rule Hidden_Archive_Signature
{
    meta:
        description = "Detects hidden archive files"
        severity = "low"
        
    strings:
        $zip = { 50 4B 03 04 }  // ZIP signature
        $rar = { 52 61 72 21 1A 07 00 }  // RAR signature
        $7z = { 37 7A BC AF 27 1C }  // 7Z signature
        
    condition:
        any of them
}
            ''',
            
            "polyglot_detection.yar": '''
/*
    Polyglot File Detection Rules
*/

rule Polyglot_PDF_Image
{
    meta:
        description = "Detects PDF/Image polyglot files"
        severity = "high"
        
    strings:
        $pdf_header = "%PDF"
        $jpeg_header = { FF D8 FF }
        $png_header = { 89 50 4E 47 }
        
    condition:
        $pdf_header and ($jpeg_header or $png_header)
}

rule Suspicious_Metadata_Length
{
    meta:
        description = "Detects unusually large metadata sections"
        severity = "medium"
        
    condition:
        filesize > 1MB 
}
            '''
        }
        
        # Write sample rules to files
        for filename, content in sample_rules.items():
            rule_path = Path(self.YARA_RULES_DIR) / filename
            if not rule_path.exists():
                try:
                    with open(rule_path, 'w') as f:
                        f.write(content)
                    print(f"Created sample YARA rule: {filename}")
                except Exception as e:
                    print(f"Error creating sample rule {filename}: {e}")
    
    def scan_with_yara(self, file_path: str) -> List[Dict[str, Any]]:
        """Scan file with YARA rules"""
        if not self.yara_rules:
            return []
        
        try:
            matches = self.yara_rules.match(file_path)
            yara_results = []
            
            for match in matches:
                # Extract metadata
                meta = {}
                if hasattr(match, 'meta'):
                    for key, value in match.meta.items():
                        meta[key] = value
                
                # Determine severity
                severity = meta.get('severity', 'medium').lower()
                if severity not in ['low', 'medium', 'high', 'critical']:
                    severity = 'medium'
                
                yara_result = {
                    'rule_name': match.rule,
                    'description': meta.get('description', f'YARA rule {match.rule} matched'),
                    'severity': severity,
                    'author': meta.get('author', 'Unknown'),
                    'meta': meta,
                    'strings_matched': []
                }
                
                # Add matched strings info
                for string in match.strings:
                    yara_result['strings_matched'].append({
                        'identifier': string.identifier,
                        'offset': string.instances[0].offset if string.instances else 0,
                        'length': string.instances[0].length if string.instances else 0
                    })
                
                yara_results.append(yara_result)
            
            return yara_results
            
        except Exception as e:
            print(f"YARA scanning error: {e}")
            return []
    
    def validate_file(self, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate file using basic parameters"""
        issues = []
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            issues.append({
                'type': 'file_size',
                'message': f'File too large: {file_size} bytes (max: {MAX_FILE_SIZE})',
                'severity': 'error'
            })
        
        if file_size < MIN_FILE_SIZE:
            issues.append({
                'type': 'file_size',
                'message': f'File too small: {file_size} bytes (min: {MIN_FILE_SIZE})',
                'severity': 'warning'
            })
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            issues.append({
                'type': 'file_extension',
                'message': f'Unsupported file extension: {file_ext}',
                'severity': 'error'
            })
        
        return {
            'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
            'issues': issues,
            'file_extension': file_ext
        }
    
    def calculate_file_hash(self, file_path: str) -> Dict[str, str]:
        """Calculate file hashes"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            for algorithm in HASH_ALGORITHMS:
                if hasattr(hashlib, algorithm):
                    hash_func = getattr(hashlib, algorithm)
                    hashes[algorithm] = hash_func(content).hexdigest()
        except Exception as e:
            print(f"Error calculating hashes: {e}")
        
        return hashes
    
    def check_malware_hash(self, file_hash: str) -> bool:
        """Check if hash matches known malware"""
        return file_hash.lower() in self.malware_hashes
    
    def get_mime_type_validation(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Basic MIME type validation without python-magic"""
        try:
            file_extension = Path(filename).suffix.lower()
            
            # Basic validation based on file signature
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            detected_type = self._detect_file_type_basic(header)
            
            # Expected types based on extension
            expected_types = {
                '.jpg': 'JPEG', '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.gif': 'GIF',
                '.bmp': 'BMP'
            }
            
            expected = expected_types.get(file_extension, 'Unknown')
            is_valid = (detected_type == expected) or (detected_type == 'Unknown')
            
            return {
                'detected_mime': f'image/{detected_type.lower()}' if detected_type != 'Unknown' else 'unknown',
                'expected_mimes': [f'image/{expected.lower()}'] if expected != 'Unknown' else [],
                'is_valid_mime': is_valid,
                'file_description': f'{detected_type} image file'
            }
        except Exception as e:
            return {
                'error': f"MIME validation error: {e}",
                'is_valid_mime': False
            }
    
    def _detect_file_type_basic(self, header: bytes) -> str:
        """Basic file type detection using magic bytes"""
        if header.startswith(b'\xFF\xD8\xFF'):
            return 'JPEG'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'PNG'
        elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            return 'GIF'
        elif header.startswith(b'BM'):
            return 'BMP'
        elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
            return 'TIFF'
        else:
            return 'Unknown'
    
    def determine_risk_level(self, scan_results: Dict[str, Any]) -> str:
        """Determine overall risk level"""
        max_risk = 0
        
        # Check malware hash matches
        if scan_results.get('hash_analysis', {}).get('known_malware', False):
            max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
        
        # Check YARA matches
        yara_matches = scan_results.get('yara_matches', [])
        for match in yara_matches:
            severity = match.get('severity', 'medium').lower()
            if severity == 'critical':
                max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
            elif severity == 'high':
                max_risk = max(max_risk, RISK_LEVELS['HIGH'])
            elif severity == 'medium':
                max_risk = max(max_risk, RISK_LEVELS['MEDIUM'])
            elif severity == 'low':
                max_risk = max(max_risk, RISK_LEVELS['LOW'])
        
        # Check EXIF threats
        exif_threats = scan_results.get('exif_analysis', {}).get('exif_threats', [])
        for threat in exif_threats:
            severity = threat.get('severity', 'low')
            if severity == 'critical':
                max_risk = max(max_risk, RISK_LEVELS['CRITICAL'])
            elif severity == 'high':
                max_risk = max(max_risk, RISK_LEVELS['HIGH'])
        
        # Check file validation issues
        validation_issues = scan_results.get('validation_issues', [])
        for issue in validation_issues:
            if issue.get('severity') == 'error':
                max_risk = max(max_risk, RISK_LEVELS['MEDIUM'])
        
        # Convert numeric risk back to string
        risk_map = {v: k for k, v in RISK_LEVELS.items()}
        return risk_map.get(max_risk, 'CLEAN')
    
    def scan_file(self, file_data: bytes, filename: str, is_full_scan: bool = False) -> Dict[str, Any]:
        """Main method to scan uploaded file - enhanced with YARA"""
        start_time = time.time()
        
        # Validate file first
        validation_result = self.validate_file(filename, len(file_data))
        if not validation_result['valid']:
            return {
                'status': 'error',
                'message': 'File validation failed',
                'validation_issues': validation_result['issues'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        try:
            # Perform enhanced scan
            if is_full_scan:
                result = self.perform_full_scan(temp_file_path, filename)
            else:
                result = self.perform_light_scan(temp_file_path, filename)
            
            # Add scan metadata
            result['scan_info'] = {
                'scan_type': 'full' if is_full_scan else 'light',
                'duration': round(time.time() - start_time, 3),
                'file_size': len(file_data),
                'filename': filename,
                'yara_enabled': YARA_AVAILABLE and self.yara_rules is not None
            }
            
            return result
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")
    
    def perform_light_scan(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Perform light scan - enhanced with YARA"""
        start_time = time.time()
        scan_results = {
            'scan_type': 'light',
            'file_info': self.get_file_info(file_path, filename),
            'hash_analysis': {},
            'yara_matches': [],
            'exif_analysis': {},
            'mime_validation': {},
            'threats_detected': 0
        }
        
        try:
            # Hash analysis - only SHA256 for light scan
            hashes = {'sha256': hashlib.sha256(open(file_path, 'rb').read()).hexdigest()}
            scan_results['hash_analysis'] = {
                'hashes': hashes,
                'known_malware': self.check_malware_hash(hashes['sha256'])
            }
            
            # YARA scanning - ENABLED!
            if YARA_AVAILABLE and self.yara_rules:
                scan_results['yara_matches'] = self.scan_with_yara(file_path)
            
            # MIME type validation
            scan_results['mime_validation'] = self.get_mime_type_validation(file_path, filename)
            
            # Basic EXIF analysis if PIL available
            if PIL_AVAILABLE:
                exif_data = self.extract_exif_data(file_path)
                exif_threats = self.analyze_exif_threats(exif_data, is_full_scan=False)
                scan_results['exif_analysis'] = {
                    'exif_data': exif_data,
                    'exif_threats': exif_threats
                }
            else:
                scan_results['exif_analysis'] = {
                    'exif_data': {},
                    'exif_threats': [],
                    'note': 'PIL not available for EXIF analysis'
                }
            
            # Count total threats
            threats_count = 0
            if scan_results['hash_analysis']['known_malware']:
                threats_count += 1
            threats_count += len(scan_results['yara_matches'])  # YARA matches now count!
            threats_count += len(scan_results['exif_analysis']['exif_threats'])
            
            scan_results['threats_detected'] = threats_count
            scan_results['risk_level'] = self.determine_risk_level(scan_results)
            scan_results['scan_duration'] = round(time.time() - start_time, 3)
            scan_results['status'] = 'success'
            
        except Exception as e:
            scan_results['status'] = 'error'
            scan_results['error'] = str(e)
            scan_results['risk_level'] = 'UNKNOWN'
        
        return scan_results
    
    def perform_full_scan(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Perform full scan - enhanced with YARA"""
        start_time = time.time()
        scan_results = {
            'scan_type': 'full',
            'file_info': self.get_file_info(file_path, filename),
            'hash_analysis': {},
            'yara_matches': [],
            'exif_analysis': {},
            'mime_validation': {},
            'entropy_analysis': {},
            'format_analysis': {},
            'threats_detected': 0
        }
        
        try:
            # Complete hash analysis
            scan_results['hash_analysis'] = {
                'hashes': self.calculate_file_hash(file_path),
                'known_malware': False
            }
            
            # Check all hashes against malware database
            for hash_type, hash_value in scan_results['hash_analysis']['hashes'].items():
                if hash_type == 'sha256' and self.check_malware_hash(hash_value):
                    scan_results['hash_analysis']['known_malware'] = True
                    break
            
            # YARA scanning - ENABLED!
            if YARA_AVAILABLE and self.yara_rules:
                scan_results['yara_matches'] = self.scan_with_yara(file_path)
            
            # Complete MIME validation
            scan_results['mime_validation'] = self.get_mime_type_validation(file_path, filename)
            
            # Complete EXIF analysis if PIL available
            if PIL_AVAILABLE:
                exif_data = self.extract_exif_data(file_path)
                exif_threats = self.analyze_exif_threats(exif_data, is_full_scan=True)
                scan_results['exif_analysis'] = {
                    'exif_data': exif_data,
                    'exif_threats': exif_threats,
                    'privacy_concerns': self._analyze_privacy_data(exif_data)
                }
            else:
                scan_results['exif_analysis'] = {
                    'exif_data': {},
                    'exif_threats': [],
                    'privacy_concerns': [],
                    'note': 'PIL not available for EXIF analysis'
                }
            
            # Basic entropy analysis
            scan_results['entropy_analysis'] = self._calculate_entropy_basic(file_path)
            
            # Basic format analysis
            scan_results['format_analysis'] = self._analyze_file_format_basic(file_path, filename)
            
            # Count total threats
            threats_count = 0
            if scan_results['hash_analysis']['known_malware']:
                threats_count += 1
            threats_count += len(scan_results['yara_matches'])  # YARA matches now count!
            threats_count += len(scan_results['exif_analysis']['exif_threats'])
            
            scan_results['threats_detected'] = threats_count
            scan_results['risk_level'] = self.determine_risk_level(scan_results)
            scan_results['scan_duration'] = round(time.time() - start_time, 3)
            scan_results['status'] = 'success'
            
        except Exception as e:
            scan_results['status'] = 'error'
            scan_results['error'] = str(e)
            scan_results['risk_level'] = 'UNKNOWN'
        
        return scan_results
    
    def get_file_info(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Get basic file information"""
        try:
            file_stat = os.stat(file_path)
            return {
                'filename': filename,
                'size_bytes': file_stat.st_size,
                'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'extension': Path(filename).suffix.lower(),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
        except Exception as e:
            return {
                'filename': filename,
                'error': f"Error getting file info: {str(e)}"
            }
    
    def extract_exif_data(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF data from image if PIL available"""
        if not PIL_AVAILABLE:
            return {}
        
        exif_data = {}
        try:
            with Image.open(file_path) as img:
                exif_dict = img._getexif()
                
                if exif_dict:
                    for tag_id, value in exif_dict.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except UnicodeDecodeError:
                                value = value.hex()
                        
                        exif_data[str(tag)] = str(value)
        except Exception as e:
            print(f"Error extracting EXIF: {e}")
        
        return exif_data
    
    def analyze_exif_threats(self, exif_data: Dict[str, Any], is_full_scan: bool) -> List[Dict[str, Any]]:
        """Analyze EXIF data for threats - enhanced version"""
        threats = []
        
        # Basic patterns to check for
        critical_patterns = ['<script', 'eval(', 'system(', 'exec(', 'shell_exec(', 'javascript:']
        
        for field, value in exif_data.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern in critical_patterns:
                    if pattern in value_lower:
                        threats.append({
                            'type': 'exif_threat',
                            'subtype': 'suspicious_content',
                            'field': field,
                            'value': value[:50],
                            'severity': 'high',
                            'description': f'Suspicious content in EXIF {field}'
                        })
        
        return threats
    
    def _analyze_privacy_data(self, exif_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze EXIF data for privacy concerns"""
        privacy_issues = []
        
        # Check for GPS data
        gps_fields = ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPSInfo']
        for field in gps_fields:
            if field in exif_data and exif_data[field]:
                privacy_issues.append({
                    'type': 'privacy_concern',
                    'subtype': 'gps_location',
                    'field': field,
                    'severity': 'low',
                    'description': 'GPS location data found'
                })
        
        return privacy_issues
    
    def _calculate_entropy_basic(self, file_path: str) -> Dict[str, Any]:
        """Basic entropy calculation"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if not data:
                return {'entropy': 0, 'analysis': 'empty_file'}
            
            # Simple entropy calculation
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            entropy = 0
            data_len = len(data)
            for count in byte_counts:
                if count > 0:
                    frequency = count / data_len
                    entropy -= frequency * (frequency.bit_length() - 1)
            
            # Normalize to 0-1 range
            entropy = entropy / 8.0 if entropy > 0 else 0
            
            if entropy > 0.75:
                analysis = 'high_entropy'
                risk = 'medium'
            elif entropy > 0.6:
                analysis = 'moderate_entropy'
                risk = 'low'
            else:
                analysis = 'normal_entropy'
                risk = 'none'
            
            return {
                'entropy': round(entropy, 3),
                'analysis': analysis,
                'risk_level': risk,
                'file_size': data_len
            }
        
        except Exception as e:
            return {'error': f'Entropy calculation failed: {e}'}
    
    def _analyze_file_format_basic(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Basic file format analysis"""
        analysis = {
            'format_consistency': True,
            'header_analysis': {},
            'issues': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)  # Read first 32 bytes
            
            file_extension = Path(filename).suffix.lower()
            detected_format = self._detect_file_type_basic(header)
            
            analysis['header_analysis'] = {
                'detected_format': detected_format,
                'expected_format': file_extension,
                'header_hex': header[:16].hex()
            }
            
            # Check for format mismatches
            expected_formats = {
                '.jpg': 'JPEG', '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.gif': 'GIF',
                '.bmp': 'BMP'
            }
            
            expected = expected_formats.get(file_extension)
            if expected and detected_format != expected and detected_format != 'Unknown':
                analysis['format_consistency'] = False
                analysis['issues'].append({
                    'type': 'format_mismatch',
                    'description': f'Extension {file_extension} does not match detected format {detected_format}',
                    'severity': 'medium'
                })
        
        except Exception as e:
            analysis['error'] = f'Format analysis failed: {e}'
        
        return analysis
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics"""
        return {
            'service_name': 'ImageSecurityService',
            'version': '3.0.0-enhanced',
            'mode': 'enhanced' if YARA_AVAILABLE and self.yara_rules else 'simplified',
            'malware_hashes_loaded': len(self.malware_hashes),
            'yara_rules_compiled': bool(self.yara_rules),
            'yara_available': YARA_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'config_available': CONFIG_AVAILABLE,
            'scan_types': ['light', 'full'],
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'risk_levels': list(RISK_LEVELS.keys()),
            'hash_algorithms': HASH_ALGORITHMS,
            'features': {
                'hash_analysis': True,
                'exif_analysis': PIL_AVAILABLE,
                'yara_scanning': YARA_AVAILABLE and bool(self.yara_rules),
                'magic_detection': False,
                'entropy_analysis': True,
                'format_analysis': True,
                'malware_hash_checking': len(self.malware_hashes) > 0
            }
        }
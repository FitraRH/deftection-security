# controllers/image_security_controller.py - Fixed Version
"""
Image Security Controller - Fixed with all required methods
"""

import json
import base64
from datetime import datetime
from typing import Dict, Any, Tuple
from flask import jsonify
from werkzeug.datastructures import FileStorage
from pathlib import Path

# Import configuration with fallback
try:
    from config import (
        MAX_FILE_SIZE,
        MIN_FILE_SIZE,
        ALLOWED_EXTENSIONS,
        RISK_LEVELS,
        ERROR_CODES,
        DEBUG_MODE,
        DETAILED_ERRORS
    )
    CONFIG_AVAILABLE = True
except ImportError:
    # Fallback configuration if config import fails
    print("Warning: Could not import config, using fallback values")
    CONFIG_AVAILABLE = False
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 100
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    RISK_LEVELS = {'CLEAN': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    ERROR_CODES = {
        'FILE_TOO_LARGE': 'E001',
        'INVALID_FILE_TYPE': 'E002',
        'INTERNAL_ERROR': 'E999'
    }
    DEBUG_MODE = True
    DETAILED_ERRORS = True

# Import service with fallback
try:
    from services.image_security_service import ImageSecurityService
    SERVICE_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ImageSecurityService")
    SERVICE_AVAILABLE = False
    ImageSecurityService = None


class ImageSecurityController:
    """Complete Image Security Controller with all required methods"""
    
    def __init__(self):
        """Initialize controller with proper error handling"""
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
        self.RISK_LEVELS = RISK_LEVELS
        self.ERROR_CODES = ERROR_CODES
        
        # Initialize service with error handling
        if SERVICE_AVAILABLE and ImageSecurityService:
            try:
                self.service = ImageSecurityService()
                print("✅ ImageSecurityService initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize ImageSecurityService: {e}")
                self.service = None
        else:
            print("❌ ImageSecurityService not available")
            self.service = None
        
        # Statistics tracking
        self.stats = {
            'scans_performed': 0,
            'threats_detected': 0,
            'files_processed': 0,
            'start_time': datetime.now(),
            'last_scan_time': None,
            'scan_types': {'light': 0, 'full': 0},
            'risk_levels': {level: 0 for level in RISK_LEVELS.keys()}
        }
    
    def health_check(self) -> Tuple[Dict[str, Any], int]:
        """Health check endpoint"""
        try:
            # Basic health check
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service_name': 'Image Security Scanner',
                'version': '2.0.0'
            }
            
            # Check service availability
            if self.service:
                try:
                    service_info = self.service.get_service_info()
                    health_status['service_info'] = service_info
                    health_status['service_status'] = 'operational'
                except Exception as e:
                    health_status['service_status'] = 'degraded'
                    health_status['service_error'] = str(e)
            else:
                health_status['service_status'] = 'not_available'
                health_status['service_error'] = 'ImageSecurityService not initialized'
            
            # Add configuration info
            health_status['configuration'] = {
                'max_file_size_mb': self.MAX_FILE_SIZE // (1024 * 1024),
                'allowed_extensions': list(self.ALLOWED_EXTENSIONS),
                'scan_types': ['light', 'full'],
                'debug_mode': DEBUG_MODE,
                'config_available': CONFIG_AVAILABLE,
                'service_available': SERVICE_AVAILABLE
            }
            
            # Determine overall status
            if health_status['service_status'] == 'operational':
                return health_status, 200
            else:
                health_status['status'] = 'degraded'
                return health_status, 200  # Still return 200 for health checks
                
        except Exception as e:
            error_response = {
                'status': 'unhealthy',
                'error': f"Health check failed: {str(e) if DETAILED_ERRORS else 'Internal error'}",
                'timestamp': datetime.now().isoformat(),
                'error_code': self.ERROR_CODES['INTERNAL_ERROR']
            }
            return error_response, 500
    
    def get_scanner_stats(self) -> Tuple[Dict[str, Any], int]:
        """Get scanner statistics - FIXED: Added missing method"""
        try:
            uptime = datetime.now() - self.stats['start_time']
            
            # Calculate rates
            total_scans = self.stats['scans_performed']
            scans_per_hour = total_scans / max(uptime.total_seconds() / 3600, 1)
            
            stats_response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'uptime': {
                        'seconds': int(uptime.total_seconds()),
                        'hours': round(uptime.total_seconds() / 3600, 2),
                        'start_time': self.stats['start_time'].isoformat()
                    },
                    'performance': {
                        'total_scans': total_scans,
                        'files_processed': self.stats['files_processed'],
                        'threats_detected': self.stats['threats_detected'],
                        'scans_per_hour': round(scans_per_hour, 2),
                        'last_scan_time': self.stats['last_scan_time']
                    },
                    'scan_types': self.stats['scan_types'].copy(),
                    'risk_distribution': self.stats['risk_levels'].copy(),
                    'service_status': {
                        'service_available': self.service is not None,
                        'config_available': CONFIG_AVAILABLE,
                        'scan_capabilities': {
                            'hash_checking': True,
                            'exif_analysis': self.service is not None,
                            'basic_validation': True,
                            'entropy_analysis': True
                        }
                    }
                }
            }
            
            # Add service-specific stats if available
            if self.service:
                try:
                    service_info = self.service.get_service_info()
                    stats_response['statistics']['service_info'] = {
                        'version': service_info.get('version', 'unknown'),
                        'mode': service_info.get('mode', 'unknown'),
                        'features': service_info.get('features', {}),
                        'malware_hashes_loaded': service_info.get('malware_hashes_loaded', 0)
                    }
                except Exception as e:
                    stats_response['statistics']['service_error'] = str(e)
            
            return stats_response, 200
            
        except Exception as e:
            error_response = {
                'status': 'error',
                'error': f"Statistics retrieval failed: {str(e) if DETAILED_ERRORS else 'Internal error'}",
                'timestamp': datetime.now().isoformat(),
                'error_code': self.ERROR_CODES['INTERNAL_ERROR']
            }
            return error_response, 500
    
    def scan_image(self, request) -> Tuple[Dict[str, Any], int]:
        """Main image scanning endpoint"""
        start_time = datetime.now()
        
        try:
            if not self.service:
                return self._error_response(
                    message="Image security service not available",
                    error_code=self.ERROR_CODES['INTERNAL_ERROR'],
                    status_code=503
                )
            
            # Parse request and validate
            file_data, filename, is_full_scan, validation_error = self._parse_scan_request(request)
            
            if validation_error:
                return validation_error
            
            print(f"Starting {'full' if is_full_scan else 'light'} scan for {filename}")
            
            # Update stats
            self._update_scan_stats(is_full_scan)
            
            # Perform scan using service
            scan_result = self.service.scan_file(file_data, filename, is_full_scan)
            
            # Update stats with results
            self._update_result_stats(scan_result)
            
            # Format response
            response = self._format_scan_response(scan_result, start_time)
            
            # Log scan completion
            risk_level = scan_result.get('risk_level', 'UNKNOWN')
            threats_count = scan_result.get('threats_detected', 0)
            duration = scan_result.get('scan_duration', 0)
            
            print(f"Scan completed: {filename} - Risk: {risk_level}, Threats: {threats_count}, Duration: {duration}s")
            
            # Return appropriate HTTP status based on risk level
            status_code = self._get_http_status_for_risk(risk_level)
            return response, status_code
        
        except Exception as e:
            print(f"Scan error: {e}")
            return self._error_response(
                message=f"Internal scan error: {str(e) if DETAILED_ERRORS else 'Processing failed'}",
                error_code=self.ERROR_CODES['INTERNAL_ERROR'],
                status_code=500
            )
    
    def scan_image_laravel(self, request) -> Tuple[Dict[str, Any], int]:
        """Laravel-compatible image scanning endpoint"""
        start_time = datetime.now()
        
        try:
            if not self.service:
                return self._error_response_laravel(
                    message="Image security service not available",
                    error_code=self.ERROR_CODES['INTERNAL_ERROR'],
                    status_code=503
                )
            
            # Parse request and validate (same as original)
            file_data, filename, is_full_scan, validation_error = self._parse_scan_request(request)
            
            if validation_error:
                # Convert error to Laravel format
                return self._error_response_laravel_from_original(validation_error)
            
            print(f"Starting {'full' if is_full_scan else 'light'} scan for {filename} (Laravel format)")
            
            # Update stats
            self._update_scan_stats(is_full_scan)
            
            # Perform scan using service (same as original)
            scan_result = self.service.scan_file(file_data, filename, is_full_scan)
            
            # Update stats with results
            self._update_result_stats(scan_result)
            
            # Format response using Laravel formatter
            response = self.format_laravel_response(scan_result, start_time, filename)
            
            # Log scan completion
            risk_level = scan_result.get('risk_level', 'UNKNOWN')
            threats_count = scan_result.get('threats_detected', 0)
            duration = scan_result.get('scan_duration', 0)
            
            print(f"Laravel scan completed: {filename} - Risk: {risk_level}, Threats: {threats_count}, Duration: {duration}s")
            
            # Return appropriate HTTP status based on risk level
            status_code = self._get_http_status_for_risk(risk_level)
            return response, status_code
        
        except Exception as e:
            print(f"Laravel scan error: {e}")
            return self._error_response_laravel(
                message=f"Internal scan error: {str(e) if DETAILED_ERRORS else 'Processing failed'}",
                error_code=self.ERROR_CODES['INTERNAL_ERROR'],
                status_code=500
            )

    def format_laravel_response(self, scan_result: Dict[str, Any], start_time: datetime, filename: str) -> Dict[str, Any]:
        """Format response for Laravel compatibility"""
        try:
            from services.security_laravel_response_service import SecurityLaravelResponseService
            return SecurityLaravelResponseService.format_response(scan_result, start_time, filename)
        except ImportError:
            # Fallback Laravel response format if service not available
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Extract basic information
            hash_analysis = scan_result.get('hash_analysis', {})
            hash_value = hash_analysis.get('hashes', {}).get('sha256', 'unknown')
            risk_level = scan_result.get('risk_level', 'CLEAN').lower()
            threats_detected = scan_result.get('threats_detected', 0)
            
            # Simple status mapping
            if threats_detected == 0:
                status = 'clean'
            elif risk_level in ['critical', 'high']:
                status = 'malicious'
            else:
                status = 'suspicious'
            
            # Basic flags
            flags = [f"{scan_result.get('scan_type', 'unknown')}_scan"]
            if threats_detected > 0:
                flags.append(f"risk_{risk_level}")
            
            return {
                'status': scan_result.get('status', 'success'),
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'hash': hash_value,
                    'status': status,
                    'risk_level': risk_level,
                    'flags': flags,
                    'details': {'threats_detected': threats_detected},
                    'possible_attack': [],
                    'processing_time_ms': round(total_duration * 1000, 3)
                }
            }

    def _update_scan_stats(self, is_full_scan: bool):
        """Update scan statistics"""
        self.stats['scans_performed'] += 1
        self.stats['files_processed'] += 1
        self.stats['last_scan_time'] = datetime.now().isoformat()
        
        if is_full_scan:
            self.stats['scan_types']['full'] += 1
        else:
            self.stats['scan_types']['light'] += 1
    
    def _update_result_stats(self, scan_result: Dict[str, Any]):
        """Update statistics with scan results"""
        # Update threat count
        threats_detected = scan_result.get('threats_detected', 0)
        if threats_detected > 0:
            self.stats['threats_detected'] += threats_detected
        
        # Update risk level distribution
        risk_level = scan_result.get('risk_level', 'CLEAN')
        if risk_level in self.stats['risk_levels']:
            self.stats['risk_levels'][risk_level] += 1

    def _error_response_laravel(self, message: str, error_code: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
        """Generate Laravel-formatted error response"""
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'error_code': error_code,
                'message': message,
                'hash': None,
                'status': 'error',
                'risk_level': 'unknown',
                'flags': ['error'],
                'details': {'error': message},
                'possible_attack': [],
                'processing_time_ms': 0
            }
        }, status_code

    def _error_response_laravel_from_original(self, original_error: Tuple[Dict[str, Any], int]) -> Tuple[Dict[str, Any], int]:
        """Convert original error response to Laravel format"""
        error_dict, status_code = original_error
        
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'error_code': error_dict.get('error_code', self.ERROR_CODES['INTERNAL_ERROR']),
                'message': error_dict.get('message', 'Unknown error'),
                'hash': None,
                'status': 'error',
                'risk_level': 'unknown',
                'flags': ['error'],
                'details': {'error': error_dict.get('message', 'Unknown error')},
                'possible_attack': [],
                'processing_time_ms': 0
            }
        }, status_code
    
    def _parse_scan_request(self, request) -> Tuple[bytes, str, bool, Dict[str, Any]]:
        """Parse and validate scan request"""
        try:
            file_data = None
            filename = None
            is_full_scan = False
            
            # Handle JSON requests (base64 encoded images)
            if request.is_json:
                json_data = request.get_json()
                
                if not json_data:
                    return None, None, False, self._error_response(
                        "Invalid JSON request",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                # Extract base64 image
                image_base64 = json_data.get('image_base64', '')
                filename = json_data.get('filename', 'uploaded_image.jpg')
                is_full_scan = json_data.get('is_full_scan', False)
                
                if not image_base64:
                    return None, None, False, self._error_response(
                        "Missing image_base64 field",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                # Decode base64 data
                try:
                    # Handle data URI format
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    
                    file_data = base64.b64decode(image_base64)
                except Exception as e:
                    return None, None, False, self._error_response(
                        f"Invalid base64 image data: {e}",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
            
            # Handle form data requests (file upload)
            else:
                if 'file' not in request.files:
                    return None, None, False, self._error_response(
                        "No file provided in request",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                file_obj: FileStorage = request.files['file']
                
                if file_obj.filename == '':
                    return None, None, False, self._error_response(
                        "No file selected",
                        self.ERROR_CODES['INVALID_FILE_TYPE'],
                        400
                    )
                
                filename = file_obj.filename
                file_data = file_obj.read()
                is_full_scan = request.form.get('is_full_scan', 'false').lower() == 'true'
            
            # Validate file
            validation_result = self._validate_file_request(file_data, filename)
            if validation_result:
                return None, None, False, validation_result
            
            return file_data, filename, is_full_scan, None
        
        except Exception as e:
            print(f"Request parsing error: {e}")
            return None, None, False, self._error_response(
                f"Request parsing failed: {str(e) if DETAILED_ERRORS else 'Invalid request'}",
                self.ERROR_CODES['INVALID_FILE_TYPE'],
                400
            )
    
    def _validate_file_request(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate file request"""
        # Check file size
        if len(file_data) > self.MAX_FILE_SIZE:
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            actual_mb = len(file_data) / (1024 * 1024)
            return self._error_response(
                f"File too large: {actual_mb:.1f}MB (max: {max_mb:.1f}MB)",
                self.ERROR_CODES['FILE_TOO_LARGE'],
                413
            )
        
        if len(file_data) < MIN_FILE_SIZE:
            return self._error_response(
                f"File too small: {len(file_data)} bytes (min: {MIN_FILE_SIZE})",
                self.ERROR_CODES['INVALID_FILE_TYPE'],
                400
            )
        
        # Check file extension
        if filename:
            file_extension = Path(filename).suffix.lower()
            
            if file_extension not in self.ALLOWED_EXTENSIONS:
                allowed_list = ', '.join(sorted(self.ALLOWED_EXTENSIONS))
                return self._error_response(
                    f"Unsupported file type: {file_extension}. Allowed: {allowed_list}",
                    self.ERROR_CODES['INVALID_FILE_TYPE'],
                    400
                )
        
        return None
    
    def _format_scan_response(self, scan_result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Format scan response"""
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Base response structure
        response = {
            'status': scan_result.get('status', 'success'),
            'timestamp': datetime.now().isoformat(),
            'scan_info': {
                'scan_type': scan_result.get('scan_type', 'unknown'),
                'duration': round(total_duration, 3),
                'risk_level': scan_result.get('risk_level', 'UNKNOWN'),
                'threats_detected': scan_result.get('threats_detected', 0)
            }
        }
        
        # Add scan results
        if scan_result.get('status') == 'success':
            response['scan_result'] = {
                'file_info': scan_result.get('file_info', {}),
                'hash_analysis': scan_result.get('hash_analysis', {}),
                'yara_matches': scan_result.get('yara_matches', []),
                'exif_analysis': scan_result.get('exif_analysis', {}),
                'mime_validation': scan_result.get('mime_validation', {}),
                'risk_assessment': {
                    'level': scan_result.get('risk_level', 'UNKNOWN'),
                    'score': self.RISK_LEVELS.get(scan_result.get('risk_level', 'CLEAN'), 0),
                    'threats_count': scan_result.get('threats_detected', 0)
                }
            }
            
            # Add full scan specific results
            if scan_result.get('scan_type') == 'full':
                response['scan_result'].update({
                    'entropy_analysis': scan_result.get('entropy_analysis', {}),
                    'format_analysis': scan_result.get('format_analysis', {})
                })
        
        return response
    
    def _get_http_status_for_risk(self, risk_level: str) -> int:
        """Get HTTP status code based on risk level"""
        risk_status_map = {
            'CLEAN': 200,
            'LOW': 200,
            'MEDIUM': 200,
            'HIGH': 200,
            'CRITICAL': 200,  # Still 200 for successful scan, just dangerous content
            'UNKNOWN': 200
        }
        return risk_status_map.get(risk_level, 200)
    
    def _error_response(self, message: str, error_code: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
        """Generate error response"""
        return {
            'status': 'error',
            'error_code': error_code,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }, status_code
    
    def get_scanner_stats(self) -> Tuple[Dict[str, Any], int]:
        """Get scanner statistics - FIXED: Added missing method"""
        try:
            # Simple stats response
            stats_response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'service_status': {
                        'service_available': self.service is not None,
                        'service_initialized': True
                    },
                    'performance': {
                        'scans_performed': 0,
                        'threats_detected': 0,
                        'uptime': 'running'
                    }
                }
            }
            
            # Add service info if available
            if self.service:
                try:
                    service_info = self.service.get_service_info()
                    stats_response['statistics']['service_info'] = service_info
                except Exception as e:
                    stats_response['statistics']['service_error'] = str(e)
            
            return stats_response, 200
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, 500


# Test function
def test_controller():
    """Test function to verify the controller works"""
    print("Testing ImageSecurityController...")
    
    try:
        # Initialize controller
        controller = ImageSecurityController()
        print("✅ Controller initialized successfully")
        
        # Test all required methods
        required_methods = [
            'health_check', 
            'scan_image', 
            'scan_image_laravel', 
            'get_scanner_stats',
            'format_laravel_response'
        ]
        
        for method_name in required_methods:
            if hasattr(controller, method_name):
                print(f"✅ {method_name} method exists")
                
                # Test health_check method specifically
                if method_name == 'health_check':
                    try:
                        health_response, status_code = controller.health_check()
                        print(f"✅ health_check method works - Status: {status_code}")
                        print(f"   Response status: {health_response.get('status')}")
                    except Exception as e:
                        print(f"❌ health_check method failed: {e}")
                
                # Test get_scanner_stats method specifically
                elif method_name == 'get_scanner_stats':
                    try:
                        stats_response, status_code = controller.get_scanner_stats()
                        print(f"✅ get_scanner_stats method works - Status: {status_code}")
                        print(f"   Total scans: {stats_response.get('statistics', {}).get('performance', {}).get('total_scans', 0)}")
                    except Exception as e:
                        print(f"❌ get_scanner_stats method failed: {e}")
                        
            else:
                print(f"❌ {method_name} method missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Controller test failed: {e}")
        return False


if __name__ == "__main__":
    test_controller()
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
import json
import tempfile
from datetime import datetime
from werkzeug.datastructures import FileStorage
from io import BytesIO

from controllers.detection_controller import DetectionController
from services.detection_service import DetectionService
from controllers.image_security_controller import ImageSecurityController

class EnhancedAPIServer:
    """API Server: Flask-AI + Security Scanner + Real-time Frame Processing"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        
        self._setup_logging()
        
        self.detection_service = DetectionService()
        self.detection_controller = DetectionController(self.detection_service)
        self.security_controller = ImageSecurityController()
        
        self._setup_routes()
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup all API endpoints"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.detection_controller.health_check()
        
        @self.app.route('/api/system/info', methods=['GET'])
        def system_info():
            return self.detection_controller.get_system_info()
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            return self.detection_controller.get_system_status()
        
        @self.app.route('/api/detection/image', methods=['POST'])
        def detect_image():
            return self.detection_controller.process_image(request)
        
        @self.app.route('/api/detection/frame', methods=['POST'])
        def detect_frame():
            """Real-time frame processing with enhanced detection"""
            return self.detection_controller.process_frame(request)
        
        @self.app.route('/api/detection/batch', methods=['POST'])
        def detect_batch():
            return self.detection_controller.process_batch(request)
        
        @self.app.route('/api/config/thresholds', methods=['GET'])
        def get_thresholds():
            return self.detection_controller.get_detection_thresholds()
        
        @self.app.route('/api/config/thresholds', methods=['PUT'])
        def update_thresholds():
            return self.detection_controller.update_detection_thresholds(request)
        
        @self.app.route('/api/config/reset', methods=['PUT'])
        def reset_thresholds():
            return self.detection_controller.reset_detection_thresholds(request)
        
        @self.app.route('/api/detection/combined', methods=['POST'])
        def detect_combined():
            """Combined defect detection + security scan"""
            try:
                defect_result = self.detection_controller.process_image(request)
                
                if hasattr(defect_result, 'get_json'):
                    defect_data = defect_result.get_json()
                    defect_status_code = defect_result.status_code
                elif isinstance(defect_result, tuple):
                    defect_response, defect_status_code = defect_result
                    if hasattr(defect_response, 'get_json'):
                        defect_data = defect_response.get_json()
                    else:
                        defect_data = defect_response.json if hasattr(defect_response, 'json') else defect_response
                else:
                    defect_data = defect_result
                    defect_status_code = 200
                
                is_scan_threat = True
                
                if request.is_json or 'application/json' in str(request.content_type):
                    json_data = request.get_json()
                    if json_data:
                        is_scan_threat = json_data.get('is_scan_threat', True)
                elif request.form:
                    is_scan_threat = request.form.get('is_scan_threat', 'true').lower() == 'true'
                
                if is_scan_threat:
                    try:
                        security_result = self._perform_security_scan_with_proper_data(request)
                        
                        if isinstance(security_result, tuple):
                            security_data, security_status_code = security_result
                        else:
                            security_data = security_result
                            security_status_code = 200
                        
                        if defect_status_code == 200 and security_status_code == 200:
                            if not isinstance(defect_data, dict):
                                defect_data = {'data': defect_data} if defect_data else {'data': {}}
                            
                            defect_data['security_scan'] = security_data.get('data', security_data)
                            defect_data['combined_analysis'] = True
                            defect_data['timestamp'] = datetime.now().isoformat()
                            
                            return jsonify(defect_data), 200
                        else:
                            if not isinstance(defect_data, dict):
                                defect_data = {'data': defect_data} if defect_data else {'data': {}}
                            
                            defect_data['security_scan'] = {
                                'status': 'error',
                                'error': 'Security scan failed',
                                'details': security_data if security_status_code != 200 else 'Unknown error'
                            }
                            defect_data['combined_analysis'] = True
                            defect_data['timestamp'] = datetime.now().isoformat()
                            
                            return jsonify(defect_data), defect_status_code
                    
                    except Exception as security_error:
                        self.logger.error(f"Security scan error: {security_error}")
                        if not isinstance(defect_data, dict):
                            defect_data = {'data': defect_data} if defect_data else {'data': {}}
                        
                        defect_data['security_scan'] = {
                            'status': 'error',
                            'error': f'Security scan failed: {str(security_error)}',
                            'details': str(security_error)
                        }
                        defect_data['combined_analysis'] = True
                        defect_data['timestamp'] = datetime.now().isoformat()
                        
                        return jsonify(defect_data), defect_status_code
                else:
                    if not isinstance(defect_data, dict):
                        defect_data = {'data': defect_data} if defect_data else {'data': {}}
                    
                    defect_data['combined_analysis'] = False
                    defect_data['timestamp'] = datetime.now().isoformat()
                    
                    return jsonify(defect_data), defect_status_code
                    
            except Exception as e:
                self.logger.error(f"Combined detection error: {e}")
                return jsonify({
                    'status': 'error',
                    'error': f'Combined detection failed: {str(e)}',
                    'timestamp': datetime.now().isoformat(),
                    'combined_analysis': False
                }), 500
        
        @self.app.route('/api/security/scan', methods=['POST'])
        def security_scan():
            """Security scan endpoint (normal format)"""
            return self.security_controller.scan_image(request)
        
        @self.app.route('/api/security/scan/laravel', methods=['POST'])
        def security_scan_laravel():
            """Security scan endpoint (Laravel format)"""
            return self.security_controller.scan_image_laravel(request)
        
        @self.app.route('/api/security/health', methods=['GET'])
        def security_health():
            return self.security_controller.health_check()
        
        @self.app.route('/api/security/stats', methods=['GET'])
        def security_stats():
            return self.security_controller.get_scanner_stats()
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An internal error occurred while processing the request',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad request',
                'message': 'Invalid request data or parameters',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        @self.app.errorhandler(413)
        def file_too_large(error):
            return jsonify({
                'error': 'File too large',
                'message': 'File exceeds maximum allowed size',
                'timestamp': datetime.now().isoformat()
            }), 413
        
        @self.app.errorhandler(415)
        def unsupported_media_type(error):
            return jsonify({
                'error': 'Unsupported Media Type',
                'message': 'Content-Type not supported. Use application/json or multipart/form-data',
                'supported_types': ['application/json', 'multipart/form-data'],
                'timestamp': datetime.now().isoformat()
            }), 415
    
    def _perform_security_scan_with_proper_data(self, original_request):
        """Perform security scan with proper image data transfer"""
        try:
            image_data = None
            filename = None
            
            if original_request.is_json or 'application/json' in str(original_request.content_type):
                json_data = original_request.get_json()
                if json_data:
                    image_base64 = (json_data.get('image_base64') or 
                                  json_data.get('image') or 
                                  json_data.get('file_base64') or
                                  json_data.get('data'))
                    
                    filename = json_data.get('filename', 'security_scan.jpg')
                    
                    if image_base64:
                        if isinstance(image_base64, str) and ',' in image_base64:
                            image_base64 = image_base64.split(',')[1]
                        
                        import base64
                        image_data = base64.b64decode(image_base64)
            
            elif original_request.files:
                for field_name in ['image', 'file', 'upload', 'data']:
                    if field_name in original_request.files:
                        file_obj = original_request.files[field_name]
                        if file_obj.filename != '':
                            file_obj.seek(0)
                            image_data = file_obj.read()
                            filename = file_obj.filename or 'security_scan.jpg'
                            file_obj.seek(0)
                            break
            
            if not image_data:
                return {
                    'status': 'error',
                    'data': {
                        'error_code': 'E002',
                        'message': 'No image data found for security scan',
                        'details': {'error': 'No image data available'},
                        'status': 'error'
                    }
                }, 400
            
            from io import BytesIO
            from werkzeug.datastructures import FileStorage
            
            file_stream = BytesIO(image_data)
            file_storage = FileStorage(
                stream=file_stream,
                filename=filename,
                content_type='image/jpeg'
            )
            
            class DirectSecurityRequest:
                def __init__(self, file_storage, filename):
                    self.files = {'image': file_storage}
                    self.form = {'is_full_scan': 'false'}
                    self.content_type = 'multipart/form-data'
                    self.is_json = False
                    self.method = 'POST'
                
                def get_json(self):
                    return None
            
            direct_request = DirectSecurityRequest(file_storage, filename)
            
            result = self.security_controller.scan_image_laravel(direct_request)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in security scan with proper data: {e}")
            return {
                'status': 'error',
                'data': {
                    'error_code': 'E999',
                    'message': f'Security scan setup failed: {str(e)}',
                    'details': {'error': str(e)},
                    'status': 'error'
                }
            }, 500
    
    def run(self, debug=False):
        """Start the API server"""
        print("Starting Enhanced API Server")
        print("=" * 50)
        print(f"Server URL: http://{self.host}:{self.port}")
        print()
        print("FLASK-AI ENDPOINTS:")
        print(f"  Health Check: GET /api/health")
        print(f"  System Info:  GET /api/system/info") 
        print(f"  Detect Image: POST /api/detection/image")
        print(f"  Detect Frame: POST /api/detection/frame")
        print(f"  Batch Detect: POST /api/detection/batch")
        print()
        print("COMBINED ENDPOINT:")
        print(f"  Combined:     POST /api/detection/combined")
        print(f"                Param: is_scan_threat=true for security scan")
        print()
        print("SECURITY SCANNER ENDPOINTS:")
        print(f"  Security Scan: POST /api/security/scan")
        print(f"  Laravel Format: POST /api/security/scan/laravel")
        print(f"  Security Health: GET /api/security/health")
        print(f"  Security Stats:  GET /api/security/stats")
        print("=" * 50)
        print("Features:")
        print("  - Real-time frame processing")
        print("  - Product-aware detection")
        print("  - OpenAI analysis integration")
        print("  - Security scanning")
        print("  - Combined detection + security")
        print("  - Background class filtering")
        print("  - False positive protection")
        print("=" * 50)
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)


def create_enhanced_api_server(host='0.0.0.0', port=5001):
    """Factory function to create API server"""
    return EnhancedAPIServer(host=host, port=port)


if __name__ == "__main__":
    api_server = create_enhanced_api_server()
    api_server.run(debug=True)
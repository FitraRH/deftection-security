# api_server.py - Combined Flask-AI + Security Scanner
"""
Combined Stateless API Server for Defect Detection + Security Scanning
Main base: flask-ai defect detection system
Added: Security scanning capabilities from pythonsec2
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
from datetime import datetime

# Import controllers from flask-ai (main base)
from controllers.detection_controller import DetectionController

# Import services from flask-ai (main base) 
from services.detection_service import DetectionService

# Import security scanning components from pythonsec2
from controllers.image_security_controller import ImageSecurityController

class CombinedAPIServer:
    """
    Combined API Server: Flask-AI (main) + Security Scanner
    """
    
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        
        # Setup logging
        self._setup_logging()
        
        # Initialize services (flask-ai main base)
        self.detection_service = DetectionService()
        
        # Initialize controllers (flask-ai main base)
        self.detection_controller = DetectionController(self.detection_service)
        
        # Initialize security scanner (from pythonsec2)
        self.security_controller = ImageSecurityController()
        
        # Setup routes
        self._setup_routes()
        
        print("Combined API Server initialized (Flask-AI + Security Scanner)")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('combined_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup all API endpoints"""
        
        # ===========================
        # FLASK-AI ROUTES (MAIN BASE)
        # ===========================
        
        # Health and system endpoints (flask-ai)
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.detection_controller.health_check()
        
        @self.app.route('/api/system/info', methods=['GET'])
        def system_info():
            return self.detection_controller.get_system_info()
        
        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            return self.detection_controller.get_system_status()
        
        # Detection endpoints (flask-ai)
        @self.app.route('/api/detection/image', methods=['POST'])
        def detect_image():
            return self.detection_controller.process_image(request)
        
        @self.app.route('/api/detection/frame', methods=['POST'])
        def detect_frame():
            return self.detection_controller.process_frame(request)
        
        @self.app.route('/api/detection/batch', methods=['POST'])
        def detect_batch():
            return self.detection_controller.process_batch(request)
        
        # Configuration endpoints (flask-ai)
        @self.app.route('/api/config/thresholds', methods=['GET'])
        def get_thresholds():
            return self.detection_controller.get_detection_thresholds()
        
        @self.app.route('/api/config/thresholds', methods=['PUT'])
        def update_thresholds():
            return self.detection_controller.update_detection_thresholds(request)
        
        # ===========================
        # NEW COMBINED ENDPOINT
        # ===========================
        
        @self.app.route('/api/detection/combined', methods=['POST'])
        def detect_combined():
            """
            NEW: Combined defect detection + security scan
            Parameter: is_scan_threat (boolean) - if true, adds security scan with is_full_scan=false
            """
            try:
                # Get defect detection result (flask-ai main)
                defect_result = self.detection_controller.process_image(request)
                
                # Check if security scan requested
                is_scan_threat = False
                if request.json:
                    is_scan_threat = request.json.get('is_scan_threat', False)
                elif request.form:
                    is_scan_threat = request.form.get('is_scan_threat', 'false').lower() == 'true'
                
                if is_scan_threat:
                    # Add security scan with is_full_scan=false (light scan)
                    # Modify request to set is_full_scan=false for security scan
                    if hasattr(request, 'json') and request.json:
                        request.json['is_full_scan'] = False
                    elif hasattr(request, 'form') and request.form:
                        # For form data, we can't modify, but security scanner will default to false
                        pass
                    
                    security_result = self.security_controller.scan_image_laravel(request)
                    
                    # Merge responses
                    if defect_result[1] == 200 and security_result[1] == 200:
                        defect_data = defect_result[0]
                        security_data = security_result[0]
                        
                        # Add security scan to defect result
                        defect_data['data']['security_scan'] = security_data.get('data', {})
                        defect_data['data']['combined_analysis'] = True
                        
                        return defect_data, 200
                    else:
                        # Return defect result even if security scan fails
                        return defect_result
                else:
                    # Return only defect detection
                    return defect_result
                    
            except Exception as e:
                self.logger.error(f"Combined detection error: {e}")
                return jsonify({
                    'status': 'error',
                    'error': f'Combined detection failed: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        # ===========================
        # SECURITY SCANNER ENDPOINTS
        # ===========================
        
        @self.app.route('/api/security/scan', methods=['POST'])
        def security_scan():
            """
            NEW: Security scan endpoint (normal format)
            Parameter: is_full_scan (boolean) - from request
            """
            return self.security_controller.scan_image(request)
        
        @self.app.route('/api/security/scan/laravel', methods=['POST'])
        def security_scan_laravel():
            """
            NEW: Security scan endpoint (Laravel format)
            Parameter: is_full_scan (boolean) - from request
            """
            return self.security_controller.scan_image_laravel(request)
        
        # Security health endpoints
        @self.app.route('/api/security/health', methods=['GET'])
        def security_health():
            return self.security_controller.health_check()
        
        @self.app.route('/api/security/stats', methods=['GET'])
        def security_stats():
            return self.security_controller.get_scanner_stats()
        
        # ===========================
        # ERROR HANDLERS
        # ===========================
        
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
    
    def run(self, debug=False):
        """Start the Combined API server"""
        print("Starting Combined API Server (Flask-AI + Security Scanner)")
        print("=" * 60)
        print(f"Server URL: http://{self.host}:{self.port}")
        print()
        print("FLASK-AI ENDPOINTS (Main Base):")
        print(f"  Health Check: GET /api/health")
        print(f"  System Info:  GET /api/system/info") 
        print(f"  Detect Image: POST /api/detection/image")
        print(f"  Detect Frame: POST /api/detection/frame")
        print(f"  Batch Detect: POST /api/detection/batch")
        print()
        print("NEW COMBINED ENDPOINT:")
        print(f"  Combined:     POST /api/detection/combined")
        print(f"                (param: is_scan_threat=true for security scan)")
        print()
        print("SECURITY SCANNER ENDPOINTS:")
        print(f"  Security Scan: POST /api/security/scan")
        print(f"                 (param: is_full_scan=true/false)")
        print(f"  Laravel Format: POST /api/security/scan/laravel")
        print(f"                  (param: is_full_scan=true/false)")
        print(f"  Security Health: GET /api/security/health")
        print("=" * 60)
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)


def create_combined_api_server(host='0.0.0.0', port=5001):
    """Factory function to create combined API server"""
    return CombinedAPIServer(host=host, port=port)


if __name__ == "__main__":
    # Create and start the combined API server
    api_server = create_combined_api_server()
    api_server.run(debug=True)
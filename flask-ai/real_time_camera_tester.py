#!/usr/bin/env python3
"""
Real-time Camera Tester for Enhanced Frame Processing API
Test the upgraded /api/detection/frame endpoint with live camera feed
"""

import cv2
import requests
import base64
import json
import time
import threading
from datetime import datetime
import argparse
import sys

class RealTimeCameraTester:
    def __init__(self, api_url="http://localhost:5001", camera_id=0):
        self.api_url = api_url.rstrip('/')
        self.camera_id = camera_id
        self.frame_endpoint = f"{self.api_url}/api/detection/frame"
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'defects_detected': 0,
            'good_frames': 0,
            'total_processing_time': 0,
            'start_time': time.time()
        }
        
        # Settings
        self.frame_interval = 1.0  # Process every 1 second
        self.show_full_response = False
        self.save_annotated = False
        self.running = True
        
        print("=" * 70)
        print("Real-time Camera Tester for Enhanced Frame Processing")
        print("=" * 70)
        print(f"API URL: {self.api_url}")
        print(f"Frame Endpoint: {self.frame_endpoint}")
        print(f"Camera ID: {camera_id}")
        print("=" * 70)
    
    def test_api_connection(self):
        """Test API server connection"""
        try:
            health_url = f"{self.api_url}/api/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Server: {health_data.get('status', 'unknown')}")
                
                # Check if real-time capabilities are available
                capabilities = health_data.get('real_time_capabilities', {})
                if capabilities.get('enhanced_frame_processing'):
                    print("✅ Enhanced Frame Processing: Available")
                    print(f"✅ Smart Processing: {capabilities.get('smart_processing', False)}")
                    print(f"✅ Adaptive Thresholds: {capabilities.get('adaptive_thresholds', False)}")
                    print(f"✅ Guaranteed Defect Detection: {capabilities.get('guaranteed_defect_detection', False)}")
                else:
                    print("⚠️  Enhanced Frame Processing: Not detected")
                
                return True
            else:
                print(f"❌ API Server not responding: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API server: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"❌ Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Cannot read from camera")
                return False
            
            print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def capture_and_encode_frame(self):
        """Capture frame and encode to base64"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return frame, frame_base64
            
        except Exception as e:
            print(f"❌ Frame capture failed: {e}")
            return None, None
    
    def send_frame_for_processing(self, frame_base64, frame_number):
        """Send frame to enhanced frame processing endpoint"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_frame_{timestamp}_{frame_number}.jpg"
            
            # Prepare request payload
            payload = {
                "frame_base64": frame_base64,
                "filename": filename,
                "fast_mode": True,
                "include_annotation": True
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(
                self.frame_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.stats['successful_requests'] += 1
                self.stats['total_processing_time'] += processing_time
                
                # Update statistics
                final_decision = result.get('final_decision', 'UNKNOWN')
                if final_decision == 'DEFECT':
                    self.stats['defects_detected'] += 1
                elif final_decision == 'GOOD':
                    self.stats['good_frames'] += 1
                
                return result, processing_time
            else:
                self.stats['failed_requests'] += 1
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None, processing_time
                
        except requests.exceptions.RequestException as e:
            self.stats['failed_requests'] += 1
            print(f"❌ Request failed: {e}")
            return None, 0
    
    def display_frame_result(self, frame, result, processing_time, frame_number):
        """Display frame with processing results"""
        if result is None:
            return frame
        
        try:
            # Get detection info
            final_decision = result.get('final_decision', 'UNKNOWN')
            anomaly_score = result.get('anomaly_score', 0.0)
            defects = result.get('defects', [])
            
            # Color based on decision
            if final_decision == 'GOOD':
                color = (0, 255, 0)  # Green
                border_color = (0, 255, 0)
            elif final_decision == 'DEFECT':
                color = (0, 0, 255)  # Red
                border_color = (0, 0, 255)
            else:
                color = (0, 255, 255)  # Yellow
                border_color = (0, 255, 255)
            
            # Add border
            frame_with_border = cv2.copyMakeBorder(
                frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color
            )
            
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Decision text
            cv2.putText(frame_with_border, f"Decision: {final_decision}", 
                       (20, 40), font, 1.2, color, 3)
            
            # Score text
            cv2.putText(frame_with_border, f"Score: {anomaly_score:.3f}", 
                       (20, 80), font, 0.8, (255, 255, 255), 2)
            
            # Processing time
            cv2.putText(frame_with_border, f"Time: {processing_time:.3f}s", 
                       (20, 120), font, 0.8, (255, 255, 255), 2)
            
            # Frame number
            cv2.putText(frame_with_border, f"Frame: {frame_number}", 
                       (20, 160), font, 0.8, (255, 255, 255), 2)
            
            # Defect count
            if defects:
                cv2.putText(frame_with_border, f"Defects: {len(defects)}", 
                           (20, 200), font, 0.8, (255, 255, 0), 2)
                
                # Draw bounding boxes for defects
                for i, defect in enumerate(defects[:3]):  # Show max 3 defects
                    bbox = defect.get('bounding_box', {})
                    if bbox:
                        x, y = bbox.get('x', 0), bbox.get('y', 0)
                        w, h = bbox.get('width', 0), bbox.get('height', 0)
                        
                        # Adjust coordinates for border
                        x += 10
                        y += 10
                        
                        # Draw rectangle
                        cv2.rectangle(frame_with_border, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        
                        # Add label
                        label = defect.get('label', 'defect')
                        cv2.putText(frame_with_border, label.upper(), (x, y - 5),
                                   font, 0.5, (0, 255, 255), 1)
            
            # Enhanced processing info
            if result.get('enhanced_detection'):
                cv2.putText(frame_with_border, "Enhanced: ON", 
                           (20, frame_with_border.shape[0] - 60), font, 0.6, (0, 255, 0), 2)
            
            if result.get('smart_processing'):
                cv2.putText(frame_with_border, "Smart: ON", 
                           (20, frame_with_border.shape[0] - 30), font, 0.6, (0, 255, 0), 2)
            
            return frame_with_border
            
        except Exception as e:
            print(f"❌ Display error: {e}")
            return frame
    
    def print_json_response(self, result, frame_number, processing_time):
        """Print formatted JSON response"""
        print("\n" + "=" * 70)
        print(f"FRAME #{frame_number} - Processing Time: {processing_time:.3f}s")
        print("=" * 70)
        
        if result is None:
            print("❌ No response received")
            return
        
        # Print key information
        final_decision = result.get('final_decision', 'UNKNOWN')
        anomaly_score = result.get('anomaly_score', 0.0)
        defects = result.get('defects', [])
        
        print(f"🎯 FINAL DECISION: {final_decision}")
        print(f"📊 ANOMALY SCORE: {anomaly_score:.3f}")
        print(f"🔍 DEFECTS FOUND: {len(defects)}")
        
        # Print enhanced processing info
        if result.get('enhanced_detection'):
            print("✅ Enhanced Detection: ACTIVE")
        if result.get('smart_processing'):
            print("✅ Smart Processing: ACTIVE")
        if result.get('guaranteed_defect_detection'):
            print("✅ Guaranteed Defect Detection: ACTIVE")
        
        # Print defects details
        if defects:
            print("\n🔴 DEFECTS DETECTED:")
            for i, defect in enumerate(defects, 1):
                label = defect.get('label', 'unknown')
                confidence = defect.get('confidence_score', 0.0)
                severity = defect.get('severity_level', 'unknown')
                area = defect.get('area_percentage', 0.0)
                
                print(f"  {i}. {label.upper()}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Severity: {severity}")
                print(f"     Area: {area:.2f}%")
                
                if defect.get('confidence_boosted'):
                    print("     🚀 Confidence Boosted!")
        
        # Print processing metadata
        metadata = result.get('processing_metadata', {})
        if metadata:
            print(f"\n📈 PROCESSING METADATA:")
            if metadata.get('cached_result'):
                print(f"   📦 Cached Result (Good Frames: {metadata.get('consecutive_good_count', 0)})")
            
            smart_decision = metadata.get('smart_decision', {})
            if smart_decision:
                reason = smart_decision.get('reason', 'unknown')
                print(f"   🧠 Decision Reason: {reason}")
        
        # Print full JSON if requested
        if self.show_full_response:
            print(f"\n📋 FULL JSON RESPONSE:")
            print(json.dumps(result, indent=2))
    
    def print_statistics(self):
        """Print current statistics"""
        elapsed_time = time.time() - self.stats['start_time']
        
        print("\n" + "=" * 70)
        print("📊 REAL-TIME PROCESSING STATISTICS")
        print("=" * 70)
        print(f"⏱️  Runtime: {elapsed_time:.1f}s")
        print(f"🎥 Total Frames: {self.stats['total_frames']}")
        print(f"✅ Successful: {self.stats['successful_requests']}")
        print(f"❌ Failed: {self.stats['failed_requests']}")
        print(f"🔴 Defects: {self.stats['defects_detected']}")
        print(f"🟢 Good: {self.stats['good_frames']}")
        
        if self.stats['successful_requests'] > 0:
            avg_processing = self.stats['total_processing_time'] / self.stats['successful_requests']
            print(f"⚡ Avg Processing: {avg_processing:.3f}s")
            
            if elapsed_time > 0:
                fps = self.stats['total_frames'] / elapsed_time
                print(f"📹 Processing FPS: {fps:.2f}")
        
        if self.stats['total_frames'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_frames']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
            
            if self.stats['successful_requests'] > 0:
                defect_rate = (self.stats['defects_detected'] / self.stats['successful_requests']) * 100
                print(f"🔍 Defect Rate: {defect_rate:.1f}%")
    
    def run(self):
        """Main test loop"""
        if not self.test_api_connection():
            return False
        
        if not self.initialize_camera():
            return False
        
        print("\n🚀 Starting real-time camera testing...")
        print("Press 'q' to quit, 's' to show statistics, 'f' to toggle full JSON")
        print("=" * 70)
        
        frame_number = 0
        last_process_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Capture frame
                frame, frame_base64 = self.capture_and_encode_frame()
                if frame is None:
                    continue
                
                frame_number += 1
                self.stats['total_frames'] = frame_number
                
                # Process frame at specified interval
                if current_time - last_process_time >= self.frame_interval:
                    result, processing_time = self.send_frame_for_processing(frame_base64, frame_number)
                    
                    # Print JSON response
                    self.print_json_response(result, frame_number, processing_time)
                    
                    last_process_time = current_time
                else:
                    result = None
                    processing_time = 0
                
                # Display frame with results
                display_frame = self.display_frame_result(frame, result, processing_time, frame_number)
                cv2.imshow('Real-time Enhanced Frame Processing', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('f'):
                    self.show_full_response = not self.show_full_response
                    print(f"Full JSON response: {'ON' if self.show_full_response else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            self.print_statistics()
            print("\n✅ Camera testing completed")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Real-time Camera Tester for Enhanced Frame Processing')
    parser.add_argument('--api-url', default='http://localhost:5001', 
                       help='API server URL (default: http://localhost:5001)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera ID (default: 0)')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Frame processing interval in seconds (default: 1.0)')
    parser.add_argument('--full-json', action='store_true', 
                       help='Show full JSON responses')
    
    args = parser.parse_args()
    
    # Create tester
    tester = RealTimeCameraTester(api_url=args.api_url, camera_id=args.camera)
    tester.frame_interval = args.interval
    tester.show_full_response = args.full_json
    
    # Run test
    success = tester.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
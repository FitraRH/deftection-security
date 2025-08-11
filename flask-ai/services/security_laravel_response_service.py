"""
Security Laravel Response Service
Handles formatting security scan results for Laravel compatibility
"""

import hashlib
from datetime import datetime
from typing import Dict, Any, List

class SecurityLaravelResponseService:
    
    @staticmethod
    def format_response(scan_result: Dict[str, Any], start_time: datetime, filename: str) -> Dict[str, Any]:
        """Format scan response for Laravel compatibility"""
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Extract hash (prefer SHA256, fallback to filename hash)
        hash_value = SecurityLaravelResponseService._extract_hash(scan_result, filename)
        
        # Map status from risk level  
        status = SecurityLaravelResponseService._map_status(scan_result)
        
        # Convert risk level to lowercase
        risk_level = scan_result.get('risk_level', 'CLEAN').lower()
        
        # Extract flags (threat types, rule names, etc.)
        flags = SecurityLaravelResponseService._extract_flags(scan_result)
        
        # Extract key-value threat details
        details = SecurityLaravelResponseService._extract_details(scan_result)
        
        # Possible attacks (descriptions + recommendations)
        possible_attack = SecurityLaravelResponseService._extract_possible_attacks(scan_result)
        
        # Convert processing time to milliseconds
        processing_time_ms = round(total_duration * 1000, 3)
        
        return {
            'status': scan_result.get('status', 'success'),
            'timestamp': datetime.now().isoformat(),
            'data': {
                'hash': hash_value,
                'status': status,
                'risk_level': risk_level,
                'flags': flags,
                'details': details,
                'possible_attack': possible_attack,
                'processing_time_ms': processing_time_ms
            }
        }

    @staticmethod
    def _extract_hash(scan_result: Dict[str, Any], filename: str) -> str:
        """Extract SHA256 hash from scan result"""
        hash_analysis = scan_result.get('hash_analysis', {})
        
        # Try to get SHA256 from hash analysis
        if isinstance(hash_analysis.get('hashes'), dict) and 'sha256' in hash_analysis['hashes']:
            return hash_analysis['hashes']['sha256']
        elif 'sha256' in hash_analysis:
            return hash_analysis['sha256']
        
        # Fallback: generate hash from filename
        return hashlib.sha256(filename.encode()).hexdigest()

    @staticmethod
    def _map_status(scan_result: Dict[str, Any]) -> str:
        """Map risk level to status"""
        risk_level = scan_result.get('risk_level', 'CLEAN')
        threats_detected = scan_result.get('threats_detected', 0)
        
        if threats_detected == 0:
            return 'clean'
        elif risk_level in ['CRITICAL', 'HIGH']:
            return 'malicious'
        elif risk_level in ['MEDIUM', 'LOW']:
            return 'suspicious'
        else:
            return 'clean'

    @staticmethod
    def _extract_flags(scan_result: Dict[str, Any]) -> List[str]:
        """Extract flags from scan result"""
        flags = []
        
        # Add scan type
        scan_type = scan_result.get('scan_type', 'unknown')
        flags.append(f"{scan_type}_scan")
        
        # YARA detection flags
        yara_matches = scan_result.get('yara_matches', [])
        for match in yara_matches:
            severity = match.get('severity', 'unknown')
            flags.append(f"yara_{severity}")
            rule_name = match.get('rule_name', '').lower().replace(' ', '_')
            if rule_name:
                flags.append(rule_name)
        
        # EXIF threat flags
        exif_analysis = scan_result.get('exif_analysis', {})
        exif_threats = exif_analysis.get('exif_threats', [])
        for threat in exif_threats:
            threat_type = threat.get('type', '').replace(' ', '_')
            flags.append(f"exif_{threat_type}")
        
        # Hash reputation flags
        hash_analysis = scan_result.get('hash_analysis', {})
        if hash_analysis.get('known_malware', False):
            flags.append('known_malware')
        
        reputation = hash_analysis.get('reputation', '')
        if reputation and reputation != 'unknown':
            flags.append(f"reputation_{reputation}")
        
        # Entropy flags (full scan)
        entropy_analysis = scan_result.get('entropy_analysis', {})
        if entropy_analysis.get('high_entropy', False):
            flags.append('high_entropy')
        
        # Format analysis flags (full scan)
        format_analysis = scan_result.get('format_analysis', {})
        if format_analysis.get('format_mismatch', False):
            flags.append('format_mismatch')
        
        # Risk level flag
        risk_level = scan_result.get('risk_level', '').lower()
        if risk_level:
            flags.append(f"risk_{risk_level}")
        
        return list(set(flags))  # Remove duplicates

    @staticmethod
    def _extract_details(scan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key-value threat analysis details"""
        details = {}
        
        # Hash match details
        hash_analysis = scan_result.get('hash_analysis', {})
        if hash_analysis.get('known_malware', False):
            details['match'] = "SHA256 matched entry in malware_hashes.txt"
        elif hash_analysis.get('reputation') == 'malicious':
            details['match'] = "Hash flagged as malicious by reputation service"
        
        # YARA indicators
        yara_matches = scan_result.get('yara_matches', [])
        if yara_matches:
            indicators = []
            for match in yara_matches:
                rule_name = match.get('rule_name', '').upper()
                description = match.get('description', '')
                
                if 'NOP' in rule_name or 'SLED' in rule_name:
                    indicators.append("NOP sled detected")
                elif 'XOR' in rule_name or 'LOOP' in rule_name:
                    indicators.append("XOR loop detected")
                elif 'WEB_SHELL' in rule_name or 'BACKDOOR' in rule_name:
                    indicators.append("Web shell pattern detected")
                elif 'MALWARE' in rule_name:
                    indicators.append("Malware signature detected")
                elif 'STEGANOGRAPHY' in rule_name or 'STEG' in rule_name:
                    indicators.append("Steganography tools detected")
                elif description:
                    indicators.append(description)
            
            if indicators:
                details['yara_indicators'] = ", ".join(indicators)
        
        # EXIF analysis
        exif_analysis = scan_result.get('exif_analysis', {})
        exif_threats = exif_analysis.get('exif_threats', [])
        
        for threat in exif_threats:
            threat_type = threat.get('type', '')
            threat_value = threat.get('value', '')
            
            if 'http://' in threat_value or 'https://' in threat_value:
                details['qr_content'] = threat_value
                if threat_value.startswith('http://'):
                    details['qr_protocol'] = "Unencrypted URL found in QR code"
            
            if threat_type == 'gps_coordinates':
                details['privacy_leak'] = f"GPS coordinates found: {threat_value}"
            elif threat_type == 'metadata_exposure':
                details['metadata_exposure'] = threat.get('description', 'Sensitive metadata found')
            elif 'phishing' in threat.get('description', '').lower():
                details['domain_analysis'] = "Domain resembles known phishing patterns"
        
        # Entropy analysis (full scan)
        entropy_analysis = scan_result.get('entropy_analysis', {})
        if entropy_analysis.get('high_entropy', False):
            entropy_score = entropy_analysis.get('entropy_score', 0)
            details['entropy'] = f"Entropy of {entropy_score:.2f} suggests encryption or packed data"
        
        # Format analysis (full scan)
        format_analysis = scan_result.get('format_analysis', {})
        if format_analysis.get('format_mismatch', False):
            expected = format_analysis.get('expected_format', '')
            actual = format_analysis.get('actual_format', '')
            details['format_mismatch'] = f"Expected {expected} but found {actual}"
        
        # MIME validation
        mime_validation = scan_result.get('mime_validation', {})
        if not mime_validation.get('valid', True):
            details['mime_validation'] = mime_validation.get('reason', 'MIME type validation failed')
        
        return details

    @staticmethod
    def _extract_possible_attacks(scan_result: Dict[str, Any]) -> List[str]:
        """Extract possible attacks and recommendations"""
        attacks = []
        
        # YARA threat descriptions
        yara_matches = scan_result.get('yara_matches', [])
        for match in yara_matches:
            description = match.get('description', '')
            if description:
                attacks.append(description)
        
        # EXIF threat descriptions
        exif_analysis = scan_result.get('exif_analysis', {})
        exif_threats = exif_analysis.get('exif_threats', [])
        for threat in exif_threats:
            description = threat.get('description', '')
            if description:
                attacks.append(description)
        
        # Risk-based recommendations
        risk_level = scan_result.get('risk_level', 'CLEAN')
        if risk_level == 'CRITICAL':
            attacks.extend([
                "CRITICAL: Do not open or execute this file",
                "CRITICAL: Quarantine this file immediately",
                "CRITICAL: Run full system antivirus scan"
            ])
        elif risk_level == 'HIGH':
            attacks.extend([
                "HIGH RISK: Exercise extreme caution",
                "Isolate file and verify source"
            ])
        elif risk_level == 'MEDIUM':
            attacks.append("MEDIUM RISK: Review detected threats")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_attacks = []
        for attack in attacks:
            if attack not in seen:
                seen.add(attack)
                unique_attacks.append(attack)
        
        return unique_attacks
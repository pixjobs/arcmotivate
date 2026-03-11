import time
import threading
from typing import Dict, List

class Shield:
    """
    Application-level protection against LLM token exhaustion attacks and spam bots.
    Handles IP rate limiting and stealthy fast-firing session detection.
    """
    def __init__(self):
        self.ip_rate_limits: Dict[str, List[float]] = {}
        self.user_timestamps: Dict[str, List[float]] = {}
        self.ip_lock = threading.Lock()
        self.user_lock = threading.Lock()
        
        # Configuration
        self.max_chars = 1000
        self.max_requests_per_minute = 24
        self.spam_detection_messages = 5
        self.spam_detection_seconds = 20
        self.max_turns_for_spammer = 30
        
    def check_input_length(self, text: str) -> str:
        """Truncates input to prevent Context Window Flooding attacks."""
        if not text:
            return ""
        if len(text) > self.max_chars:
            return text[:self.max_chars]
        return text
        
    def check_rate_limit(self, ip: str) -> bool:
        """
        Enforces a requests/minute sliding window per IP. 
        Returns True if allowed, False if limit exceeded.
        """
        if not ip or ip == "unknown":
            return True # Cannot reliably rate limit "unknown"
            
        current_time = time.time()
        with self.ip_lock:
            times = self.ip_rate_limits.get(ip, [])
            times = [t for t in times if current_time - t < 60]
            if len(times) >= self.max_requests_per_minute:
                return False
            times.append(current_time)
            self.ip_rate_limits[ip] = times
        return True
        
    def should_halt_spammer(self, session_id: str, current_turn_count: int) -> bool:
        """
        Tracks if a session is firing messages suspiciously fast (protects against Recursive Context Expansion).
        Returns True if the session has exhibited bot behavior AND reached the max turn cap.
        """
        if not session_id:
            return False
            
        current_time = time.time()
        is_spammer = False
        
        with self.user_lock:
            times = self.user_timestamps.get(session_id, [])
            times.append(current_time)
            if len(times) > self.spam_detection_messages:
                times = times[-self.spam_detection_messages:]
            self.user_timestamps[session_id] = times
            
            if len(times) == self.spam_detection_messages:
                if times[-1] - times[0] < self.spam_detection_seconds:
                    is_spammer = True
                    
        if is_spammer and current_turn_count >= self.max_turns_for_spammer:
            return True
            
        return False

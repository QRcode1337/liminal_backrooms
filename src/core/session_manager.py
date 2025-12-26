import json
import os
from datetime import datetime

class SessionManager:
    """Manages saving and loading of conversation sessions"""

    def __init__(self, sessions_dir="sessions"):
        self.sessions_dir = sessions_dir
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)

    def save_session(self, filename, conversation, branch_conversations=None, active_branch=None, metadata=None):
        """Save the current session to a JSON file"""
        if not filename.endswith('.json'):
            filename += '.json'

        filepath = os.path.join(self.sessions_dir, filename)

        data = {
            "conversation": conversation,
            "branch_conversations": branch_conversations or {},
            "active_branch": active_branch,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
            "version": "1.0"
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True, filepath
        except Exception as e:
            return False, str(e)

    def load_session(self, filename):
        """Load a session from a JSON file"""
        filepath = os.path.join(self.sessions_dir, filename)

        if not os.path.exists(filepath):
            return False, f"File not found: {filename}"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return True, data
        except Exception as e:
            return False, str(e)

    def list_sessions(self):
        """List available saved sessions"""
        try:
            files = [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')]
            # Sort by modification time, newest first
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.sessions_dir, x)), reverse=True)
            return files
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []

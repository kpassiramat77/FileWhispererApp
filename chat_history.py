class ChatHistory:
    """Manage chat history."""
    
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.messages.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear all messages from the chat history."""
        self.messages = []
    
    def get_messages(self):
        """Get all messages in a list of tuples format."""
        return [(msg["role"], msg["content"]) for msg in self.messages]
    
    def get_langchain_format(self):
        """Get messages in LangChain format."""
        return [(msg["role"], msg["content"]) for msg in self.messages] 
"""
Tool Parser: Converts Postman collections to tool definitions
"""
import json
import os
from typing import List, Dict, Any
from pathlib import Path

class ToolParser:
    def __init__(self, collections_dir: str):
        self.collections_dir = collections_dir
    
    def parse_all_collections(self) -> List[Dict[str, Any]]:
        """Parse all Postman collections in directory"""
        tools = []
        
        collection_files = Path(self.collections_dir).glob("*.json")
        
        for file_path in collection_files:
            try:
                with open(file_path, 'r') as f:
                    collection = json.load(f)
                
                collection_tools = self._parse_collection(collection)
                tools.extend(collection_tools)
                
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
        
        return tools
    
    def _parse_collection(self, collection: Dict) -> List[Dict[str, Any]]:
        """Parse a single Postman collection"""
        tools = []
        collection_name = collection.get("info", {}).get("name", "Unknown")
        
        items = collection.get("item", [])
        
        for item in items:
            if "request" in item:
                tool = self._parse_request(item, collection_name)
                tools.append(tool)
            elif "item" in item:
                # Nested folder
                tools.extend(self._parse_folder(item, collection_name))
        
        return tools
    
    def _parse_folder(self, folder: Dict, collection_name: str) -> List[Dict]:
        """Parse nested folder"""
        tools = []
        for item in folder.get("item", []):
            if "request" in item:
                tool = self._parse_request(item, collection_name)
                tools.append(tool)
        return tools
    
    def _parse_request(self, item: Dict, collection_name: str) -> Dict[str, Any]:
        """Parse a single API request into tool format"""
        request = item.get("request", {})
        
        tool = {
            "id": self._generate_tool_id(item["name"]),
            "name": item.get("name", "Unnamed Tool"),
            "description": item.get("description", ""),
            "collection": collection_name,
            "method": request.get("method", "GET"),
            "url": self._extract_url(request),
            "headers": self._extract_headers(request),
            "parameters": self._extract_parameters(request),
            "body_type": self._get_body_type(request),
            "domain": self._infer_domain(item["name"], collection_name)
        }
        
        return tool
    
    def _generate_tool_id(self, name: str) -> str:
        """Generate unique tool ID"""
        return name.lower().replace(" ", "_").replace("-", "_")
    
    def _extract_url(self, request: Dict) -> str:
        """Extract URL from request"""
        url = request.get("url", {})
        if isinstance(url, str):
            return url
        elif isinstance(url, dict):
            raw = url.get("raw", "")
            return raw
        return ""
    
    def _extract_headers(self, request: Dict) -> List[Dict]:
        """Extract headers"""
        headers = request.get("header", [])
        return [{"key": h.get("key"), "value": h.get("value")} for h in headers]
    
    def _extract_parameters(self, request: Dict) -> List[Dict]:
        """Extract query parameters and body parameters"""
        params = []
        url = request.get("url", {})
        if isinstance(url, dict):
            query = url.get("query", [])
            for q in query:
                params.append({"name": q.get("key"), "type": "query"})
        
        body = request.get("body", {})
        if body.get("mode") == "formdata":
            for item in body.get("formdata", []):
                params.append({"name": item.get("key"), "type": "formdata"})
        
        return params
    
    def _get_body_type(self, request: Dict) -> str:
        """Get request body type"""
        body = request.get("body", {})
        return body.get("mode", "none")
    
    def _infer_domain(self, name: str, collection: str) -> str:
        """Infer domain from name and collection"""
        text = (name + " " + collection).lower()
        
        if any(word in text for word in ["invoice", "bill", "billing"]):
            return "INVOICING"
        elif any(word in text for word in ["payment", "pay", "charge", "refund"]):
            return "PAYMENTS"
        elif any(word in text for word in ["report", "analytics", "export"]):
            return "REPORTING"
        elif any(word in text for word in ["dispute", "chargeback", "claim"]):
            return "DISPUTES"
        elif any(word in text for word in ["user", "account", "profile"]):
            return "USER_MANAGEMENT"
        else:
            return "GENERAL"
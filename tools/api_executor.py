"""
API Executor: Executes API calls based on tool definitions
Handles all HTTP methods, parameter injection, and response processing
"""
import requests
from typing import Dict, Any, Optional, List
from config import tool_config
import json
import re
from urllib.parse import urlencode

class APIExecutor:
    def __init__(self):
        self.timeout = tool_config.TOOL_TIMEOUT_SECONDS
        self.session = requests.Session()
        
    def execute(self, tool_id: str, parameters: Dict[str, Any], 
                tool_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an API call based on tool definition and parameters
        
        Args:
            tool_id: Tool identifier
            parameters: Parameters extracted by the agent
            tool_definition: Complete tool metadata from indexer
            
        Returns:
            {
                "success": bool,
                "status_code": int,
                "data": dict,
                "error": str (if failed)
            }
        """
        try:
            # Build request components
            method = tool_definition["method"].upper()
            url = self._build_url(tool_definition["url"], parameters)
            headers = self._build_headers(tool_definition.get("headers", []), parameters)
            
            # Prepare request based on method
            request_kwargs = {
                "url": url,
                "headers": headers,
                "timeout": self.timeout
            }
            
            # Add body for POST/PUT/PATCH
            if method in ["POST", "PUT", "PATCH"]:
                body = self._build_body(parameters, tool_definition)
                if tool_definition.get("body_type") == "raw":
                    request_kwargs["json"] = body
                elif tool_definition.get("body_type") == "formdata":
                    request_kwargs["data"] = body
                else:
                    request_kwargs["json"] = body
            
            # Add query parameters for GET
            elif method == "GET":
                query_params = self._extract_query_params(parameters, tool_definition)
                if query_params:
                    request_kwargs["params"] = query_params
            
            # Execute request
            response = self.session.request(method, **request_kwargs)
            
            # Parse response
            return self._parse_response(response)
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status_code": 408,
                "error": f"Request timeout after {self.timeout} seconds"
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "status_code": 503,
                "error": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 500,
                "error": f"Execution failed: {str(e)}"
            }
    
    def _build_url(self, url_template: str, parameters: Dict[str, Any]) -> str:
        """
        Build URL by replacing path variables and adding query parameters
        
        Handles:
        - {variable} style: /api/users/{user_id}
        - :variable style: /api/users/:user_id
        - {{variable}} style: /api/users/{{user_id}}
        """
        url = url_template
        
        # Replace path variables
        for key, value in parameters.items():
            # Handle different variable formats
            patterns = [
                f"{{{{{key}}}}}",  # {{variable}}
                f"{{{key}}}",       # {variable}
                f":{key}"           # :variable
            ]
            
            for pattern in patterns:
                if pattern in url:
                    url = url.replace(pattern, str(value))
        
        return url
    
    def _build_headers(self, header_list: List[Dict], parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Build headers dictionary with defaults and parameter injection
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LangGraph-Agentic-System/1.0"
        }
        
        # Add headers from tool definition
        for header in header_list:
            key = header.get("key", "")
            value = header.get("value", "")
            
            if key and value:
                # Replace variables in header values
                for param_key, param_value in parameters.items():
                    value = value.replace(f"{{{param_key}}}", str(param_value))
                    value = value.replace(f"{{{{{param_key}}}}}", str(param_value))
                
                headers[key] = value
        
        # Check for auth parameters
        if "api_key" in parameters:
            headers["Authorization"] = f"Bearer {parameters['api_key']}"
        elif "auth_token" in parameters:
            headers["Authorization"] = f"Bearer {parameters['auth_token']}"
        
        return headers
    
    def _build_body(self, parameters: Dict[str, Any], tool_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build request body from parameters
        """
        body = {}
        
        # Get parameter definitions
        param_defs = tool_definition.get("parameters", [])
        body_params = [p for p in param_defs if p.get("type") in ["body", "formdata"]]
        
        if body_params:
            # Only include parameters defined for body
            for param_def in body_params:
                param_name = param_def.get("name")
                if param_name in parameters:
                    body[param_name] = parameters[param_name]
        else:
            # Include all non-path parameters
            path_params = [p["name"] for p in param_defs if p.get("type") == "path"]
            query_params = [p["name"] for p in param_defs if p.get("type") == "query"]
            
            for key, value in parameters.items():
                if key not in path_params and key not in query_params:
                    body[key] = value
        
        return body
    
    def _extract_query_params(self, parameters: Dict[str, Any], 
                             tool_definition: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract query parameters from parameters dict
        """
        query_params = {}
        
        param_defs = tool_definition.get("parameters", [])
        query_param_names = [p["name"] for p in param_defs if p.get("type") == "query"]
        
        for name in query_param_names:
            if name in parameters:
                query_params[name] = str(parameters[name])
        
        return query_params
    
    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse HTTP response into standardized format
        """
        result = {
            "success": response.status_code < 400,
            "status_code": response.status_code,
        }
        
        # Try to parse JSON response
        try:
            result["data"] = response.json()
        except json.JSONDecodeError:
            result["data"] = {"raw_response": response.text}
        
        # Add error message for failures
        if not result["success"]:
            result["error"] = self._get_error_message(response)
        
        return result
    
    def _get_error_message(self, response: requests.Response) -> str:
        """
        Extract meaningful error message from response
        """
        status_messages = {
            400: "Bad Request - Invalid parameters",
            401: "Unauthorized - Authentication required",
            403: "Forbidden - Access denied",
            404: "Not Found - Resource doesn't exist",
            429: "Rate Limit Exceeded - Too many requests",
            500: "Internal Server Error",
            503: "Service Unavailable"
        }
        
        default_msg = status_messages.get(
            response.status_code, 
            f"Request failed with status {response.status_code}"
        )
        
        # Try to get error from response body
        try:
            error_data = response.json()
            if "error" in error_data:
                return f"{default_msg}: {error_data['error']}"
            elif "message" in error_data:
                return f"{default_msg}: {error_data['message']}"
        except:
            pass
        
        return default_msg
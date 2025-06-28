#!/usr/bin/env python3
"""
Simple test script to verify Ollama integration with LOLLMS Server
"""
import requests
import json

def test_ollama_generation():
    """Test AI generation using the local Ollama model"""
    
    # Server endpoint
    url = "http://localhost:9601/api/v1/generate"
    
    # Test request
    payload = {
        "input_data": [
            {
                "type": "text",
                "role": "user_prompt",
                "data": "Hello! Please tell me a short joke about programming."
            }
        ],
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🚀 Testing LOLLMS Server with Ollama...")
    print(f"📡 Sending request to: {url}")
    print(f"💬 Prompt: {payload['input_data'][0]['data']}")
    print("\n⏳ Waiting for response...\n")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! AI Response received:")
            print("=" * 50)
            
            # Extract the text response
            if 'output' in result and len(result['output']) > 0:
                ai_response = result['output'][0].get('data', 'No response data')
                print(f"🤖 AI: {ai_response}")
            else:
                print("📄 Full response:")
                print(json.dumps(result, indent=2))
                
            print("=" * 50)
            print(f"🆔 Request ID: {result.get('request_id', 'N/A')}")
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the LOLLMS Server is running on http://localhost:9601")

if __name__ == "__main__":
    test_ollama_generation()

"""
Test script for nearest restaurants MCP tool
"""

import asyncio

from src.abstract.config_container import ConfigContainer
from src.clients.ollama_client import OllamaMCPClient

async def test_nearest_restaurants():
    """Test the find_nearest_restaurants tool"""
    
    # Sample user credentials - replace with actual token
    test_token = "your_test_token_here"
    
    # Jakarta coordinates (example)
    test_latitude = -6.195140
    test_longitude = 106.823311
    
    print("🧪 Testing Nearest Restaurants Tool")
    print("=" * 50)
    
    # Load configuration
    config = ConfigContainer.from_file("examples/server.json")
    
    # Create client
    async with await OllamaMCPClient.create(config) as client:
        print(f"📍 Testing location: {test_latitude}, {test_longitude}")
        print(f"🔍 Searching within 5km radius...")
        print()
        
        # Test 1: Ask for nearest restaurants
        print("Test 1: Natural language query")
        print("-" * 50)
        
        await client.prepare_prompt()
        
        query = f"Cari restoran terdekat dengan lokasi saya di latitude {test_latitude} dan longitude {test_longitude}"
        
        print(f"User: {query}")
        print()
        print("Assistant response:")
        
        async for response in client.process_message(query, token=test_token):
            if response["role"] == "assistant" and response["content"]:
                print(response["content"], end="", flush=True)
            elif response["role"] == "status":
                print(f"\n[{response['tool_status']}] {response['tool_name']}")
            elif response["role"] == "tool":
                print(f"\n[Tool Result] {response['content'][:200]}...")
        
        print("\n")
        print("=" * 50)
        print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_nearest_restaurants())

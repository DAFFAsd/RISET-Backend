# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "mcp[cli]",
# ]
# ///

import os
from typing import Any, Dict, List

import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("foodbot")

# Base URL for the Next.js API
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")


@mcp.tool()
async def topup_saldo(amount: float, token: str) -> Dict[str, Any]:
    """Top up user's balance/saldo
    
    Args:
        amount (float): Amount to top up (in Rupiah)
        token (str): User authentication token
    
    Returns:
        dict: Response with success status, message, and new balance
    """
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE_URL}/api/wallet/topup",
            json={"amount": amount},
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def get_restaurants(token: str) -> Dict[str, Any]:
    """Get list of all available restaurants
    
    Args:
        token (str): User authentication token
    
    Returns:
        dict: Response with list of restaurants
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/restaurants",
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def find_nearest_restaurants(
    latitude: float, 
    longitude: float, 
    token: str, 
    radius: int = 5000,
    keyword: str = ""
) -> Dict[str, Any]:
    """Find nearest restaurants or any eating places based on user's location using Google Places API.
    Returns maximum 10 nearest places sorted by distance.
    
    Args:
        latitude (float): User's latitude coordinate
        longitude (float): User's longitude coordinate
        token (str): User authentication token
        radius (int): Search radius in meters (default: 5000m or 5km)
        keyword (str): Optional search keyword to filter specific type of restaurants or food (e.g., "pizza", "chinese food", "cafe", etc.)
    
    Returns:
        dict: Response with list of nearby restaurants sorted by distance, includes:
            - success: boolean
            - restaurants: list of restaurant objects with name, distance, rating, etc.
            - totalFound: number of restaurants found (max 10)
            - keyword: search keyword used (if any)
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius
    }
    
    if keyword:
        params["keyword"] = keyword
    
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/restaurants/nearby",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def get_menu(restaurant_id: int, token: str) -> Dict[str, Any]:
    """Get menu items from a specific restaurant
    
    Args:
        restaurant_id (int): ID of the restaurant
        token (str): User authentication token
    
    Returns:
        dict: Response with list of menu items
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/restaurants/{restaurant_id}/menu",
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def create_order(
    restaurant_id: int,
    items: List[Dict[str, Any]],
    delivery_address: str,
    token: str
) -> Dict[str, Any]:
    """Create a new food order
    
    Args:
        restaurant_id (int): ID of the restaurant
        items (list): List of items with menuItemId and quantity, e.g. [{"menuItemId": 1, "quantity": 2}]
        delivery_address (str): Delivery address for the order
        token (str): User authentication token
    
    Returns:
        dict: Response with order details and updated balance
    """
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE_URL}/api/orders",
            json={
                "restaurantId": restaurant_id,
                "items": items,
                "deliveryAddress": delivery_address
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def get_my_orders(token: str) -> Dict[str, Any]:
    """Get user's order history
    
    Args:
        token (str): User authentication token
    
    Returns:
        dict: Response with list of orders
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/orders",
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def get_my_balance(token: str) -> Dict[str, Any]:
    """Get user's current balance and profile information
    
    Args:
        token (str): User authentication token
    
    Returns:
        dict: Response with user data including balance
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


@mcp.tool()
async def get_transaction_history(token: str) -> Dict[str, Any]:
    """Get user's transaction history (top ups, payments, refunds)
    
    Args:
        token (str): User authentication token
    
    Returns:
        dict: Response with list of transactions
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE_URL}/api/wallet/transactions",
            headers={"Authorization": f"Bearer {token}"}
        )
        res.raise_for_status()
        return res.json()


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")

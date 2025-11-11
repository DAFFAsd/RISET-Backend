import json
import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from itertools import chain
from typing import AsyncIterator, Optional, Self, Sequence, cast

import colorlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from ollama import AsyncClient, Message, Tool

from ..abstract.api_response import ChatResponse
from ..abstract.config_container import ConfigContainer
from ..abstract.session import Session

SYSTEM_PROMPT = """Kamu adalah FoodBot, asisten chatbot untuk aplikasi pemesanan makanan seperti Go-Food.
Kamu membantu pengguna untuk:
1. Top up saldo mereka
2. Melihat daftar restoran
3. Mencari restoran terdekat berdasarkan lokasi pengguna
4. Melihat menu dari restoran tertentu
5. Memesan makanan
6. Melihat riwayat pesanan
7. Cek saldo terkini

Kamu harus bersikap ramah, natural, dan membantu. Jangan terlalu menunjukkan bahwa kamu menggunakan tools.
Berikan respons dalam bahasa Indonesia yang santai dan natural.

Ketika user ingin top up, tanyakan berapa nominal yang ingin ditambahkan.
Ketika user ingin cari restoran terdekat, gunakan tool find_nearest_restaurants dengan koordinat lokasi mereka.
Ketika user ingin pesan makanan, tunjukkan daftar restoran terlebih dahulu, lalu menu yang tersedia.
Pastikan selalu menggunakan tools yang tersedia untuk mendapatkan informasi real-time.

# PENTING: Konteks Lokasi
- Pesan user mungkin mengandung [KONTEKS LOKASI PENGGUNA: latitude=xxx, longitude=xxx]
- Jika user meminta restoran terdekat dan konteks lokasi tersedia, LANGSUNG gunakan koordinat tersebut
- JANGAN pernah minta user memberikan koordinat secara manual jika konteks lokasi sudah ada
- Ekstrak nilai latitude dan longitude dari konteks untuk digunakan di tool find_nearest_restaurants

# Format Respons Daftar Restoren dengan Gambar:
- Ketika menampilkan restoran, SELALU sertakan gambar jika tersedia (imageUrl tidak null)
- Format markdown untuk gambar: ![Nama Restoran](url_gambar)
- Contoh format yang baik untuk find_nearest_restaurants:

**1. Nama Restoran** ⭐ 4.5
![Nama Restoran](https://maps.googleapis.com/...)
📍 120 m dari lokasimu
🏠 Alamat restoran

- Gambar harus ditampilkan SETELAH nama restoran dan rating
- Pastikan setiap restoran yang punya imageUrl ditampilkan gambarnya
- Jangan tampilkan gambar jika imageUrl adalah null

# Notes

- Selalu gunakan tools untuk mendapatkan data terbaru
- Berikan respons yang informatif dan ramah
- Format nominal uang dalam Rupiah (Rp)
- Format jarak dalam meter (m) atau kilometer (km)
- Konfirmasi pesanan sebelum memproses pembayaran
- JANGAN tampilkan [KONTEKS LOKASI PENGGUNA] dalam respons ke user
- WAJIB tampilkan gambar restoran dengan format markdown ![alt](url) jika tersedia"""


class OllamaMCPClient(AbstractAsyncContextManager):
    def __init__(self, host: Optional[str] = None):
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        console_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

        # Initialize client objects
        self.client = AsyncClient(host)
        self.servers: dict[str, Session] = {}
        self.selected_server: dict[str, Session] = {}
        self.messages = []
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            await self.exit_stack.aclose()
        except ValueError:
            return

    @classmethod
    async def create(cls, config: ConfigContainer, host: Optional[str] = None) -> Self:
        """Factory method to create and initialize a client instance"""
        client = cls(host)
        await client._connect_to_multiple_servers(config)
        return client

    async def _connect_to_multiple_servers(self, config: ConfigContainer):
        for name, params in config.stdio.items():
            session, tools = await self._connect_client(name, stdio_client(params))
            self.servers[name] = Session(session=session, tools=[*tools])
        for name, params in config.sse.items():
            session, tools = await self._connect_client(name, sse_client(**params.model_dump()))
            self.servers[name] = Session(session=session, tools=[*tools])
        for name, params in config.streamable.items():
            session, tools = await self._connect_client(name, streamablehttp_client(**params.model_dump()))
            self.servers[name] = Session(session=session, tools=[*tools])

        # Default to select all
        self.selected_server = self.servers

        self.logger.info(
            f"Connected to server with tools: {[cast(Tool.Function, tool.function).name for tool in self.get_tools()]}"
        )

    async def _connect_client(self, name: str, client) -> tuple[ClientSession, Sequence[Tool]]:
        """Connect to an stdio MCP server"""
        stdio, write = await self.exit_stack.enter_async_context(client)
        session = cast(ClientSession, await self.exit_stack.enter_async_context(ClientSession(stdio, write)))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = [
            Tool(
                type="function",
                function=Tool.Function(
                    name=f"{name}/{tool.name}",
                    description=tool.description,
                    parameters=cast(Tool.Function.Parameters, tool.inputSchema),
                ),
            )
            for tool in response.tools
        ]
        # for tool in response.tools:
        #     self.logger.debug(json.dumps(tool.inputSchema))
        return (session, tools)

    def get_tools(self) -> list[Tool]:
        return list(chain.from_iterable(server.tools for server in self.selected_server.values()))

    def select_server(self, servers: list[str]) -> Self:
        self.selected_server = {name: server for name, server in self.servers.items() if name in servers}
        self.logger.info(f"Selected server: {list(self.selected_server.keys())}")
        return self

    async def prepare_prompt(self):
        """Clear current message and create new one"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.user_token = ""

    async def process_message(self, message: str, model: str = "qwen3:8b", token: str = "") -> AsyncIterator[ChatResponse]:
        """Process a query using LLM and available tools"""
        self.messages.append({"role": "user", "content": message})
        self.user_token = token

        async for part in self._recursive_prompt(model, tool_chain=[]):
            yield part

    async def _recursive_prompt(self, model: str, tool_chain: list[str]) -> AsyncIterator[ChatResponse]:
        # self.logger.debug(f"message: {self.messages}")
        self.logger.debug(f"Prompting model '{model}' with {len(self.messages)} messages")
        self.logger.debug(f"Available tools: {[cast(Tool.Function, tool.function).name for tool in self.get_tools()]}")
        
        # Log the last user message for context
        user_messages = [msg for msg in self.messages if msg.get('role') == 'user']
        if user_messages:
            last_user_msg = user_messages[-1].get('content', '')
            preview = last_user_msg[:100] + "..." if len(last_user_msg) > 100 else last_user_msg
            self.logger.debug(f"User query: {preview}")
        
        stream = await self.client.chat(
            model=model,
            think=False, # ini hanya supaya bisa jauh lebih cepat merespon
            messages=self.messages,
            tools=self.get_tools(),
            stream=True,
        )

        tool_message_count = 0
        async for part in stream:
            if part.message.content:
                yield ChatResponse(
                    role="assistant",
                    content=part.message.content,
                    tool_name=None,
                    tool_status=None,
                    tool_chain=tool_chain if tool_chain else None
                )
            elif part.message.tool_calls:
                self.logger.debug(f"Calling tool: {part.message.tool_calls}")
                
                # Send status for each tool being called
                for tool_call in part.message.tool_calls:
                    tool_name = tool_call.function.name
                    new_tool_chain = tool_chain + [tool_name]
                    
                    # Send calling status
                    yield ChatResponse(
                        role="status",
                        content=f"Calling tool: {tool_name}",
                        tool_name=tool_name,
                        tool_status="calling",
                        tool_chain=new_tool_chain
                    )
                
                tool_messages = await self._tool_call(part.message.tool_calls, tool_chain)
                tool_message_count += 1
                for tool_message in tool_messages:
                    yield ChatResponse(
                        role="tool",
                        content=tool_message,
                        tool_name=None,
                        tool_status=None,
                        tool_chain=tool_chain if tool_chain else None
                    )
                    self.messages.append({"role": "tool", "content": tool_message})

        if tool_message_count > 0:
            async for part in self._recursive_prompt(model, tool_chain):
                yield part

    async def _tool_call(self, tool_calls: Sequence[Message.ToolCall], tool_chain: list[str]) -> list[str]:
        messages: list[str] = []
        for tool in tool_calls:
            split = tool.function.name.split("/")
            session = self.selected_server[split[0]].session
            tool_name = split[1]
            tool_args = tool.function.arguments

            # Add token to tool arguments if available and tool requires it
            if hasattr(self, 'user_token') and self.user_token and 'token' in str(tool_args):
                tool_args = {**tool_args, 'token': self.user_token}

            # Send processing status (this will be handled by the caller)
            # Execute tool call
            try:
                result = await session.call_tool(tool_name, dict(tool_args))
                self.logger.debug(f"Tool call result: {result.content}")
                message = f"tool: {tool.function.name}\nargs: {tool_args}\nreturn: {cast(TextContent, result.content[0]).text}"
            except Exception as e:
                self.logger.debug(f"Tool call error: {e}")
                message = f"Error in tool: {tool.function}\nargs: {tool_args}\n{e}"

            # Continue conversation with tool results
            messages.append(message)
        return messages

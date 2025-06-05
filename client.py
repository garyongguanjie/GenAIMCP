import asyncio
import json
from fastmcp import Client
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from typing import List, Dict, Any

# --- Configuration ---
# OpenAI client configuration for Ollama (assuming it's running locally)
OPENAI_BASE_URL = 'http://localhost:11434/v1/'
OPENAI_API_KEY = 'ollama'  # Placeholder for Ollama

# FastMCP client configuration
FAST_MCP_SERVER_PATH = "mcpserver/my_server.py"

# LLM model to use (e.g., 'llama3.2', 'mistral', 'gemma')
LLM_MODEL = 'qwen3:4b'

# --- Client Initialization ---
openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

mcp_client = Client(FAST_MCP_SERVER_PATH)

# --- Helper Function to Convert FastMCP Tools to OpenAI Tool Format ---
async def get_openai_compatible_tools() -> List[ChatCompletionToolParam]:
    """
    Fetches tools from the FastMCP server and converts them into
    the format expected by OpenAI's function calling API.
    """
    print("Fetching tools from FastMCP server...")
    fastmcp_tools = await mcp_client.list_tools()
    openai_tools: List[ChatCompletionToolParam] = []

    for tool in fastmcp_tools:
        # FastMCP tool.parameters is already a JSON schema dictionary,
        # which is exactly what OpenAI expects for 'function.parameters'.
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema  # This should be a JSON Schema dict
                },
            }
        )
    print(f"Discovered {len(openai_tools)} tools from FastMCP.")
    return openai_tools

# --- Main Asynchronous Function ---
async def main():
    """
    Main function to connect to FastMCP, fetch tools, and run the chat loop.
    """
    chat_history: List[Dict[str, Any]] = []

    try:
        # Establish connection to the FastMCP server
        async with mcp_client:
            print(f"FastMCP Client connected: {mcp_client.is_connected()}")

            # Get tools from FastMCP and format them for OpenAI
            available_openai_tools = await get_openai_compatible_tools()

            print("\n--- Starting Chat Session ---")
            while True:
                user_message = input("You: ")
                if user_message.lower() == 'exit':
                    break

                chat_history.append({"role": "user", "content": user_message})

                # Call the LLM with the current chat history and available tools
                try:
                    response = openai_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=chat_history,
                        tools=available_openai_tools if available_openai_tools else None,
                        tool_choice="auto", # Let the LLM decide if it wants to call a tool
                    )
                except Exception as e:
                    print(f"Error calling OpenAI API: {e}")
                    chat_history.pop() # Remove the last user message if API call fails
                    continue

                assistant_message = response.choices[0].message

                # Check if the LLM decided to call a tool
                if assistant_message.tool_calls:
                    print(f"LLM decided to call a tool: {assistant_message.tool_calls}")
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        print(f"  Tool Name: {tool_name}")
                        print(f"  Tool Arguments (string): {tool_args_str}")

                        try:
                            # Parse arguments from string to dictionary
                            tool_args = json.loads(tool_args_str)
                            print(f"  Tool Arguments (parsed): {tool_args}")

                            # Call the tool via FastMCP client
                            print(f"Calling FastMCP tool '{tool_name}' with args {tool_args}...")
                            tool_output = await mcp_client.call_tool(tool_name, tool_args)
                            print(f"Tool '{tool_name}' output: {tool_output}")

                            # Add the tool's response to the chat history
                            chat_history.append(assistant_message) # Add the LLM's tool_call message
                            chat_history.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": str(tool_output), # Convert output to string for content field
                                }
                            )

                            # Call the LLM again with the tool's output to get a final response
                            final_response = openai_client.chat.completions.create(
                                model=LLM_MODEL,
                                messages=chat_history,
                            )
                            print(f"Assistant: {final_response.choices[0].message.content}")
                            chat_history.append(final_response.choices[0].message)

                        except json.JSONDecodeError:
                            print(f"Error: LLM provided malformed JSON arguments for tool '{tool_name}': {tool_args_str}")
                            chat_history.pop() # Remove the last user message
                            chat_history.append({"role": "assistant", "content": "I encountered an error processing the tool arguments."})
                        except Exception as e:
                            print(f"Error executing tool '{tool_name}': {e}")
                            chat_history.pop() # Remove the last user message
                            chat_history.append({"role": "assistant", "content": f"I encountered an error while trying to use a tool: {e}"})
                else:
                    # If no tool call, print the LLM's direct response
                    print(f"Assistant: {assistant_message.content}")
                    chat_history.append(assistant_message)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # FastMCP client connection is automatically closed by 'async with'
        print(f"FastMCP Client connected after session: {mcp_client.is_connected()}")
        print("Chat session ended.")

if __name__ == "__main__":
    asyncio.run(main())

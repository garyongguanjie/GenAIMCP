from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool()
def multiply(a: int,b:int) -> int:
    return a*b

if __name__ == "__main__":
    mcp.run()
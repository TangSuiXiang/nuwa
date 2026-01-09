# Nuwa Documentation

Welcome to the Nuwa documentation! Nuwa is a powerful framework for creating AI agents with advanced capabilities including ReAct (Reasoning and Acting) patterns, tool integration, and conversation management.

## Getting Started

- [Quickstart Guide](quickstart.md) - Get up and running in minutes
- [Installation](README.md#installation) - How to install Nuwa

## Comprehensive Documentation

- [Full Documentation](README.md) - Complete guide to using Nuwa
- [Architecture and Design Patterns](architecture_design.md) - Deep dive into system architecture
- [API Reference](api_reference.md) - Detailed API documentation

## Core Concepts

Nuwa is built around several key concepts:

### Node Architecture
Everything in Nuwa is a `Node` that can be chained together to create complex processing pipelines.

### ReAct Pattern
The ReAct (Reasoning + Acting) pattern allows agents to think step-by-step and use tools when needed.

### Tool Integration
Easily register and use custom tools with decorator-based registration.

### Message Management
Built-in support for conversation history with pluggable storage backends.

## Key Features

- **Streaming Support**: Real-time response generation
- **MCP Integration**: Connect to Model Context Protocol endpoints
- **Custom System Prompts**: Dynamic prompt templating
- **Time Awareness**: Automatic timestamp inclusion
- **Multi-turn Conversations**: Persistent chat history
- **Error Handling**: Robust JSON parsing and error recovery

## Examples

The framework includes comprehensive test examples that demonstrate various usage patterns:

- Basic LLM interactions
- Chat with memory
- Tool integration
- MCP support
- Custom system prompts
- Streaming responses
- Time awareness
- User choice requests

## Support

For questions, issues, or feature requests, please refer to the project repository.

---

**Note**: This documentation is generated based on the current codebase. Always refer to the source code for the most up-to-date information.

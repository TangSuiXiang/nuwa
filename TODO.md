# Nuwa 命名重构任务清单

## 第一阶段：文件/模块重命名
- [x] `re_act.py` → `react_agent.py`
- [x] `qdrant.py` → `vector_store.py`
- [x] `vector.py` → `embeddings.py`
- [x] `search.py` → `web_search_tools.py`
- [x] `alarm.py` → `scheduling_tools.py`
- [x] `agent_group.py` → `multi_agent.py`
- [x] 处理空文件 `plan.py` → `_plan.py` 和 `image.py` → `_image.py`

## 第二阶段：类名优化
### 基础类
- [x] `Node` → `ProcessingNode`
- [x] `InputChunk` → `StreamChunk`
- [x] `MessagesManager` → `ConversationStorage`

### 智能体类
- [x] `OpenAI` → `LLMNode`
- [x] `ChatLLM` → `ConversationAgent`
- [x] `ReActAgent` → `ReasoningActingAgent`
- [x] `GroupAgent` → `MultiRoleAgent`

### 工具类
- [x] `ToolsManager` → `ToolRegistry`
- [x] `Function` → `ToolInvocation`

### 存储类
- [x] `QdrantMessagesManager` → `VectorBackedStorage`

## 第三阶段：方法/变量名优化
### 方法名改进
- [x] `__ainit__` → `_async_initialize`
- [x] `parse_input` → `_parse_input_data`
- [x] `generate_messages` → `_prepare_conversation_messages`
- [x] `parse_labels` → `_parse_stream_tags`
- [x] `parse_action` → `_extract_tool_calls`

### 变量名改进
- [x] `deps` → `dependencies`
- [x] `open_symbols` → `_open_tags_stack`
- [x] `cache` → `_character_buffer`
- [x] `think_io` → `_reasoning_buffer`

## 第四阶段：导入更新和文档
- [x] 更新 `__init__.py` 中的导出
- [x] 更新所有测试文件中的导入
- [x] 更新文档中的API引用（快速入门指南已更新，其他文档也已相应调整）

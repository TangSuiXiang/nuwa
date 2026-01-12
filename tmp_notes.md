# 命名问题记录

## base.py
1. 类 `InputChunk`:
   - `state` 字段使用字符串字面量 "DOING", "DONE", "END" – 可以使用枚举提升类型安全。
   - `content` 类型为 Union[str, Dict] – 变量名 `content` 可以，但类型联合可能造成歧义（字典内容未知）。

2. 类 `MessagesManager`:
   - 方法 `get_messages` 参数 `user_input: str = ""` – 如果用于语义检索，名称 `user_input` 可能不够精确，建议 `query` 或 `search_input`。
   - 方法 `save_messages` 和 `clear_messages` 命名清晰。

3. 类 `Node`:
   - 属性 `deps` 是 "dependencies" 的缩写，可能过于简短，建议全称 `dependencies`。
   - 方法 `get_dep` 名称单数，但返回的是单个节点，实际上可能返回依赖链中的第一个节点。名称可能引起歧义（是获取依赖列表还是获取最终依赖？）。建议 `get_final_dependency` 或 `get_root_dependency`。
   - `__gt__` 运算符重载用于建立依赖关系，但 `>` 符号可能不符合直觉（通常表示大于）。可以考虑使用 `>>`（但Python不支持）或显式方法 `connect_to`。

4. 类型提示：`Union[AsyncGenerator[InputChunk, Any], Dict]` – `Any` 使用可能过于宽泛，应指定具体类型。

暂时无严重冗长或数据类型错误。

## llm.py
1. 类 `OpenAI`:
   - 类名 `OpenAI` 可能过于通用，容易与官方库混淆。建议更具体，如 `OpenAINode` 或 `LLMNode`。
   - 属性 `think_open` 和 `symbols_open_close_map` 命名含义不清晰。`think_open` 可能表示“开始推理标记”，但名称像动词。建议 `reasoning_start_marker` 和 `symbols_mapping`。
   - 属性 `historical_messages` 可能被误解为“历史消息”缓存，但实际用途是存储当前会话的消息。建议 `cached_messages` 或 `session_messages`。
   - 方法 `__ainit__` 名称不符合常规。异步初始化通常命名为 `async_init` 或 `initialize`。
   - 方法 `parse_input` 返回 `Dict[str, Any]`，但 `Any` 过于宽泛。可以定义 `InputDict` 类型别名。
   - 变量 `proset_infos` 名称奇怪（可能是 “proset” 为 “prefix” 或 “prologue”?）。建议 `prefix_lines` 或 `additional_info_lines`。
   - 变量 `in_reasoning_mode` 可以，但 `reasoning_mode` 更简洁。
   - 方法 `generate_messages` 中 `proset_infos` 拼接逻辑复杂，可提取为辅助方法。

2. 函数/参数命名:
   - `with_others` 参数名含义模糊（“其他”指什么？）。建议 `additional_context` 或 `custom_prefix`。
   - `extra_body` 可以，但 `extra_params` 更常见。
   - `need_init` 布尔标志，命名可以，但 `_initialized` 反向标志更常见。

3. 类型提示:
   - 多处使用 `Optional[List[str]]` 和 `Optional[Dict[str, Any]]`，可以接受。
   - `AsyncGenerator[InputChunk, None]` 中第二个类型参数为 `None` 表示没有返回值，正确。

4. 其他:
   - 日志消息使用英文和中文混合，建议统一语言。
   - 字符串格式化使用 f-string 和 `.format` 混合，风格不一致。

## chat.py
1. 类 `ChatLLM`:
   - 类名可以，但 `ChatLLM` 是 `OpenAI` 的子类，可能引起混淆（是否支持其他模型？）。考虑 `ChatNode`。
   - 属性 `hook_tool_call` 命名可以，但类型 `Callable[["ChatLLM", Function], Awaitable[Any]]` 复杂，可定义类型别名。
   - 属性 `new_messages_idx` 名称含义模糊（“新消息索引”？）。实际用途是标记从哪个索引开始是新消息。建议 `new_messages_start_index`。
   - 属性 `tool_names` 和 `tools` 相似但不同（一个是名称列表，一个是实体列表)。可以合并或更明确命名，如 `available_tool_names` 和 `tool_entities`。
   - 方法 `__ainit__` 同样不符合常规。
   - 方法 `call_tool` 参数 `func` 类型为 `Function`，但实际是 `ChatCompletionMessageFunctionToolCallParam` 的字典。变量名 `func` 可能误导为可调用对象。建议 `tool_call` 或 `tool_invocation`。
   - 方法 `generate_messages` 覆盖父类，但未调用父类方法？实际上调用了 `super().generate_messages`，没问题。
   - 方法 `save_messages` 和 `clear_chat_history` 命名清晰。
   - 属性 `mcp` 可能是 “Model Context Protocol” 的缩写，但变量名 `mcp` 不明确。建议 `mcp_server_url` 或 `mcp_transport`。
   - 属性 `mcp_timeout` 可以。

2. 参数命名:
   - `session_id` 可以。
   - `tools_manager` 和 `messages_manager` 可以。
   - `hook_tool_call` 可以。

3. 类型提示:
   - `AsyncGenerator[InputChunk, None] | Dict[str, Any]` 使用联合类型，但 Python 3.10+ 支持 `|`，可读性好。
   - 多处使用 `Optional`。

4. 其他:
   - 日志消息混合中英文。
   - 字符串格式化不一致。

## tool.py
1. 类 `ToolParameter`:
   - 类名可以，但 `ToolParameter` 和 `ToolObjectParameter`、`ToolArrayParameter` 存在继承关系，命名清晰。
   - 属性 `type` 使用 `Literal` 限制，但值 "object", "string", "number", "boolean", "array" 是字符串，可能与其他类型系统冲突。考虑使用枚举 `ParameterType`。
   - 属性 `description` 和 `enum` 可以。

2. 类 `ToolEntity`:
   - 属性 `parameters` 类型为 `Union[ToolObjectParameter, ToolParameter]`，但默认值为 `ToolObjectParameter`。可能造成混淆：如果传递 `ToolParameter` 但期望对象？命名可以。

3. 类 `Tool`:
   - 属性 `func` 和 `entity` 命名清晰。

4. 函数 `get_tool_entity`:
   - 参数 `tool` 类型为 `MCPTool`，但变量名 `tool` 可能与 `Tool` 类混淆。建议 `mcp_tool`。
   - 内部变量 `t` 含义模糊（类型字符串）。建议 `type_str`。

5. 类 `ToolsManager`:
   - 属性 `_tools` 是内部字典，命名符合私有约定（前导下划线）。
   - 方法 `tool` 作为装饰器工厂，但名称 `tool` 可能与其返回的装饰器混淆。通常装饰器命名为 `register_tool` 或 `tool` 可以接受。
   - 方法 `call_tool` 参数 `func` 再次与 `Function` 类型混淆。建议 `tool_call`。
   - 方法 `add_tool`、`get_tool`、`list_tools`、`has_tool`、`clear_tools` 命名清晰。
   - 变量 `annotations` 可以。

6. 类型提示:
   - 多处使用 `Union` 和 `Optional`。
   - 使用 `typing._AnnotatedAlias` 是私有API，可能导致兼容性问题。建议使用 `getattr(v, "__origin__", None)` 等公共属性。

7. 其他:
   - 日志消息混合中英文。
   - 字符串格式化不一致。

## re_act.py
1. 类 `ReActAgent`:
   - 类名清晰。
   - 属性 `_new_messages_idx` 前缀下划线表示内部，但名称含义模糊（是布尔标志？）。实际用途是标识是否需要重置新消息索引。建议 `_should_reset_message_index`。
   - 属性 `cache`、`open_symbols` 命名可以，但 `open_symbols` 是一个栈，可以命名为 `symbol_stack`。
   - 属性 `thought_open`、`action_open`、`question_open`、`observation_open` 清晰。
   - 属性 `think_io`、`thought_io`、`action_io`、`question_io` 后缀 `_io` 表示 StringIO 对象，可以但 `_buffer` 可能更明确。
   - 属性 `round_idx` 可以，但 `current_round` 更清晰。
   - 方法 `parse_action` 返回 `List[Function]`，但 `Function` 类型是从 `chat` 导入的，可能造成混淆（与内置 `function` 冲突）。建议使用类型别名 `ToolCall`。
   - 方法 `parse_labels` 极其复杂，变量 `c`、`s`、`match_s` 含义不清晰。建议使用更具描述性的名称，如 `char`、`current_string`、`normalized_string`。
   - 许多局部变量如 `thought_close`、`action_close` 是动态计算的，但命名可以。
   - 方法 `run` 中有 `pre_chunk` 变量，含义是前一个块，但命名 `pre_chunk` 可以。
   - 变量 `tool_response` 可以。

2. 常量定义:
   - `CODE_BLOCK`、`DOUBLE_QUOTE` 等使用大写，符合常量约定。
   - `SYMBOLS_OPEN_CLOSE_MAP` 可以。

3. 参数命名:
   - `enable_chat_history`、`enable_selection` 清晰。
   - `answer_format` 可以。
   - `max_loop` 可以。

4. 类型提示:
   - 大量使用 `Union`、`Optional`、`List`、`Dict`。
   - `AsyncGenerator[InputChunk, None] | Dict[str, Any]` 使用联合类型。

5. 其他:
   - 日志消息混合中英文。
   - 字符串格式化不一致。
   - 代码复杂度高，建议拆分为辅助函数。

## qdrant.py
1. 类 `QdrantMessagesManager`:
   - 类名清晰。
   - 属性 `_collection_created` 布尔标志，命名可以。
   - 属性 `similarity_threshold`、`min_chunk_length` 清晰。
   - 方法 `get_embeddings` 委托给 `vector` 模块，名称可以。
   - 方法 `try_create_collection` 名称可以，但 `ensure_collection_exists` 更明确。
   - 方法 `get_messages` 中的变量 `time_messages_map` 类型为 `Dict[int, List[...]]`，但键是时间戳，名称可以。
   - 变量 `conversation_ids` 可以。
   - 方法 `_cosine_similarity` 静态方法，但使用 `np`，命名清晰。
   - 方法 `_convert_payload_to_message` 清晰。
   - 方法 `save_messages` 中的变量 `points` 可以。
   - 方法 `_process_tool_calls_message` 和 `_process_text_message` 命名清晰。
   - 方法 `_save_chunk` 清晰。

2. 参数命名:
   - `embedding_client` 类型为 `AsyncOpenAI`，但名称可能误导（不是嵌入客户端，而是OpenAI客户端）。建议 `openai_client`。
   - `https` 参数是布尔值，但名称 `https` 不明确（是否使用HTTPS）。可以，但 `use_https` 更好。
   - `vector_size` 可以。
   - `embedding_model` 可以。

3. 类型提示:
   - 使用 `Optional`、`List`、`Dict` 等。
   - 方法返回类型明确。

4. 其他:
   - 日志消息混合中英文。
   - 字符串格式化不一致。
   - 代码中有一些重复的向量计算，可优化。

## alarm.py
1. 类 `AlarmTask`:
   - 数据类命名清晰。
   - 属性 `remindee` 使用 `Literal["oneself", "user"]`，可以。
   - 属性 `alarm_id`、`time`、`reminder` 清晰。
   - 属性 `agent` 可以，但类型 `Optional[ReActAgent]` 可能过于具体（如果未来支持其他agent类型）。考虑使用 `Optional[Any]` 或泛型。
   - 属性 `callback` 可以。
   - 属性 `task` 可以。

2. 类 `AlarmManager`:
   - 类名清晰。
   - 属性 `tasks` 字典，键为 `alarm_id`，可以。
   - 属性 `_next_id` 私有，可以。
   - 属性 `_lock` 私有，可以。
   - 方法 `_generate_id` 私有，可以。
   - 方法 `set_alarm_for_oneself` 和 `set_alarm_for_user` 命名清晰，但重复代码较多。
   - 方法 `_schedule_alarm`、`_trigger_alarm`、`_wake_up_agent`、`_notify_user` 命名清晰。
   - 方法 `cancel_alarm` 和 `list_alarms` 清晰。

3. 函数 `get_alarm_tool`:
   - 名称 `get_alarm_tool` 可以。
   - 内部函数 `set_alarm` 参数 `time` 是ISO字符串，但变量名 `time` 可能与 `datetime` 对象混淆。建议 `time_str`。
   - 变量 `alarm_time` 可以。
   - 返回字典包含键 `success`、`message` 等，可以。

4. 全局变量 `alarm_manager`:
   - 单例实例，命名可以。

5. 类型提示:
   - 使用 `Optional`、`Literal`、`Callable` 等。
   - 函数签名清晰。

6. 其他:
   - 日志消息混合中英文。
   - 字符串格式化使用 f-string。
   - 代码结构良好。

## search.py
1. 函数 `get_google_search_tool`、`get_baidu_search_tool`、`get_bing_search_tool`:
   - 命名清晰，但 `get_*_search_tool` 返回 `Tool` 对象，可以。
   - 参数 `proxies` 类型为 `List[str]`，但变量名 `proxies` 可能暗示多个代理，实际只使用第一个。建议 `proxy_list` 或 `proxies` 可以。

2. 内部函数 `google_search`、`baidu_search`、`bing_search`:
   - 参数 `query` 可以。
   - 变量 `p` 是 `async_playwright` 上下文管理器，可以但 `playwright` 更清晰。
   - 变量 `browser`、`page` 可以。
   - 变量 `content`、`html` 可以。
   - 变量 `results`、`ret` 可以。

3. 变量命名:
   - `synopsis` 可能不常见（摘要），但可以接受。
   - `content_left` 是 XPath 选择的元素，可以。
   - `item` 可以。

4. 类型提示:
   - 函数缺少返回类型注解（例如 `google_search` 返回 `None` 但未标注）。
   - `get_*_search_tool` 返回 `Tool` 但未标注异步返回类型（`-> Tool` 实际上返回协程？实际返回 `Tool` 对象，但函数是异步，应标注 `-> Tool` 但调用者需等待）。建议标注 `async def ... -> Tool`。

5. 其他:
   - 日志消息混合中英文。
   - 字符串格式化使用 f-string。
   - 代码重复较多（三个工具类似），可提取通用逻辑。

## vector.py
1. 函数 `get_embeddings`:
   - 命名清晰。
   - 参数 `texts` 是字符串列表，可以。
   - 参数 `embedding_model` 可以。
   - 参数 `client` 类型为 `Optional[AsyncOpenAI]`，但变量名 `client` 通用，可能与其他客户端混淆。建议 `openai_client`。
   - 参数 `dimensions` 可以。
   - 局部变量 `response`、`d` 可以。
   - 返回值类型 `List[List[float]]` 清晰。

2. 类型提示:
   - 函数签名完整。
   - 使用 `Optional`、`List`。

3. 其他:
   - 日志消息使用英文。
   - 字符串格式化使用 f-string。
   - 函数简单，无严重问题。

## agent_group.py
1. 类 `GroupAgent`:
   - 类名清晰。
   - 属性 `role_blacklist` 可以，但命名 `blacklist` 可能被视为敏感词（某些代码规范建议避免）。可考虑 `excluded_roles` 或 `blocked_roles`。
   - 属性 `roles` 可以。
   - 方法 `parse_system_prompt` 与父类重复代码，可提取为公共方法。
   - 构造函数参数 `session_id=...` 使用省略号作为默认值，不常见。建议使用 `None` 或 `str = ""`。
   - 参数 `role_name` 和 `role_prompt` 清晰。

2. 类型提示:
   - 构造函数参数类型缺失（如 `session_id` 类型未指定）。
   - 使用 `List[str]` 可以。

3. 其他:
   - 日志消息混合中英文。
   - 字符串格式化使用 `.format`。
   - 代码重复（与父类相同提示模板）。

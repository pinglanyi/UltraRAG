# UltraRAG 框架调研报告

> 调研日期：2026-04-09  
> 调研范围：前后端全栈（`ui/frontend/main.js`、`ui/backend/pipeline_manager.py`、`src/ultrarag/client.py`）

---

## 一、框架技术栈概览

| 层 | 技术 |
|----|------|
| 后端框架 | Python + Flask (SSE 流式) + FastMCP (MCP 客户端/服务器) |
| 前端框架 | 原生 Vanilla JavaScript + Bootstrap（无 React/Vue） |
| Markdown | marked.js (v5+, GFM) |
| LaTeX | KaTeX + auto-render |
| 代码高亮 | highlight.js (GitHub theme) |
| XSS 防护 | DOMPurify |
| 开源协议 | **Apache 2.0** (版权归 OpenBMB) |

---

## 二、重点：流式通信协议与 Tag Flow

### 2.1 协议类型

UltraRAG 使用标准 **Server-Sent Events (SSE)**，MIME type `text/event-stream`，每条消息格式为：

```
data: {"type": "step_start", "name": "retriever.retriever_search", "depth": 0}\n\n
```

### 2.2 已定义的全部 Special Tags

框架**没有** `workflow_node_transition` 这个 tag，但以 `step_start` / `step_end` 对来实现等价功能：

| Tag 类型 | 触发时机 | Payload 字段 | 作用 |
|---------|---------|------------|------|
| `step_start` | 每个 pipeline step 开始前 | `name`, `depth` | 播报当前执行节点 |
| `step_end` | 每个 pipeline step 完成后 | `name`, `output`(摘要) | 标记节点结束 |
| `token` | LLM 逐 token 输出 | `content`, `step`, `is_final` | 流式输出内容 |
| `sources` | 检索工具完成后 | `data: [{id, title, content}]` | 传递引用来源 |
| `final` | 整个 pipeline 结束 | `{status, answer, is_first_turn, ...}` | 最终结果 |
| `error` | 任意异常 | `message` | 错误通知 |

**关键设计**：`token` tag 有 `is_final` 布尔字段：
- `is_final=false`：LLM 的"思考过程"内容（显示在可折叠的 thinking panel 中）
- `is_final=true`：最终答案 token（追加到主消息区，实时渲染 Markdown）

### 2.3 前后端交互方式（含代码位置）

**不是传统 Agentic Loop**，而是 **MCP Pipeline Executor + 可选 loop/branch 节点**。

#### 后端流程（`ui/backend/pipeline_manager.py:920-1010`）

```python
# 1. 创建队列桥接 async → sync
token_queue = queue.Queue()

async def token_callback(event_data):
    token_queue.put(event_data)  # 将事件放入队列

# 2. 后台线程执行 pipeline
def run_bg():
    res = session.run_chat(token_callback, dynamic_params)
    token_queue.put({"type": "final", "data": final_data})
    token_queue.put(None)  # 哨兵值，终止队列

threading.Thread(target=run_bg, daemon=True).start()

# 3. Flask 生成器 yield SSE
while True:
    item = token_queue.get()
    if item is None: break
    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
```

#### Pipeline 执行核心（`src/ultrarag/client.py:1371-1884`）

`execute_pipeline()` 递归遍历 YAML 中定义的 `steps` 列表：

```python
# client.py:1469-1471 — step_start 发射
await stream_callback({"type": "step_start", "name": current_step_name, "depth": depth})

# client.py:1752-1764 — generation 步骤的 token 逐个发射
async for token in local_service.generate_stream(**args_input):
    await stream_callback({
        "type": "token", "content": token,
        "step": step_identifier, "is_final": is_final_step
    })

# client.py:1698-1700 — retriever 步骤完成后发射 sources
await stream_callback({"type": "sources", "data": sources})

# client.py:1707-1712 — step_end 发射
await stream_callback({"type": "step_end", "name": current_step_name, "output": summary})
```

#### Pipeline YAML 结构示例

简单线性 RAG（`examples/vanilla_rag.yaml`）：
```yaml
pipeline:
- benchmark.get_data
- retriever.retriever_init
- retriever.retriever_search
- generation.generation_init
- prompt.qa_rag_boxed
- generation.generate
- evaluation.evaluate
```

模拟 Agentic Loop（`examples/search_r1.yaml`）：
```yaml
pipeline:
- benchmark.get_data
- retriever.retriever_init
- generation.generation_init
- generation.generate
- loop:
    times: 5          # 最多循环 5 次
    steps:
    - branch:
        router:
        - router.search_r1_check       # 路由判断
        branches:
          incomplete:                  # 未完成 → 继续检索
          - custom.search_r1_query_extract
          - retriever.retriever_search
          - generation.generate
          complete: []                 # 完成 → 退出循环
```

复杂多阶段 Agent（`examples/AgentCPM-Report.yaml`）：
```yaml
pipeline:
- loop:
    times: 140                         # 最多 140 轮
    steps:
    - branch:
        router:
        - router.surveycpm_state_router
        branches:
          search: [...]
          analyst-init_plan: [...]
          write: [...]
          done: []                     # 空分支 → 退出
```

#### Loop 终止机制（`src/ultrarag/client.py:236-240, 1474-1495`）

```python
# ContextVar 保证每个协程独立的循环控制状态
_loop_terminal_var: contextvars.ContextVar[List[bool]] = contextvars.ContextVar(...)

# loop block 执行
for st in range(times):
    loop_terminal[-1] = True
    loop_res = await _execute_steps(inner_steps, depth + 1, state)
    if loop_terminal[-1]:   # True = 终止条件满足
        loop_terminal.pop()
        break
```

#### 前端处理 SSE（`ui/frontend/main.js:4492-4626`）

```javascript
const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: true });

    for (const line of lines) {
        const data = JSON.parse(line.slice(6));  // 去掉 "data: " 前缀

        if (data.type === "step_start" || data.type === "step_end") {
            updateProcessUI(entryIndex, data);    // → 更新 "Show Thinking" 面板
        }
        else if (data.type === "sources") {
            allSources = allSources.concat(docs); // → 累积来源（含 id/title/content）
        }
        else if (data.type === "token") {
            if (!data.is_final) updateProcessUI(entryIndex, data);   // → thinking panel
            if (data.is_final) {
                currentText += data.content;
                let html = renderMarkdown(currentText, ...);
                html = formatCitationHtml(html, entryIndex);          // → 引用超链接化
                contentDiv.innerHTML = html;
                renderLatex(contentDiv);           // → KaTeX 渲染
                applyCodeHighlight(contentDiv);    // → highlight.js
                applyTableEnhancements(contentDiv);// → 表格美化
            }
        }
        else if (data.type === "final") {
            const renumbered = renumberCitations(finalText, allSources);
            renderSources(bubble, renumbered.sources, usedIds); // → 渲染引用卡片
        }
    }
}
```

---

## 三、富文本渲染与 Markdown 增强

### 3.1 代码高亮 + 复制

**已实现，完整度高。**

- `renderCodeBlock()` (`main.js:2024`) 生成带 Copy 按钮的代码块，每个块有唯一 ID
- Copy 按钮通过 `navigator.clipboard.writeText()` 实现，成功后有 2 秒视觉反馈
- `applyCodeHighlight()` (`main.js:2062`) 对所有 `pre code` 应用 highlight.js
- 语言类名自动设置为 `language-xxx`，支持所有 hljs 支持的语言

**优化程度**：实现干净，功能完整，无多余装饰。

### 3.2 结构化表格

**已实现。**

- `applyTableEnhancements()` (`main.js:2074`) 将所有 `<table>` 包裹在 `.table-scroll` 滚动容器中，防止溢出
- marked.js 的 GFM 模式自动解析 Markdown 表格语法
- 无单元格合并等高级特性（受限于 marked.js 本身）

### 3.3 LaTeX 渲染

**已实现，处理完整，有一个技术亮点。**

- `renderLatex()` (`main.js:2002`) 调用 KaTeX `renderMathInElement`
- 支持 `$$...$$`（display），`$...$`（inline），`\(...\)`，`\[...\]` 四种分隔符
- `throwOnError: false`，渲染失败静默降级不崩溃
- **亮点**：`protectMath()` / `restoreMath()` (`main.js:1980-2000`) 在 Markdown 解析前将 LaTeX 替换为占位符，解析后再还原，**防止 marked.js 将 `_` `*` 等符号误解析破坏 LaTeX**：

```javascript
// main.js:1980
function protectMath(text) {
    const mathBlocks = [];
    const regex = /(```[\s\S]*?```|`[^`\n]+`)|(\$\$[\s\S]*?\$\$)|(\\\[[\s\S]*?\\\])|...$/g;
    const protectedText = text.replace(regex, (match, code) => {
        if (code) return code;          // 代码块保持原样
        mathBlocks.push(match);
        return `MATHPLACEHOLDER${mathBlocks.length - 1}END`;  // 占位
    });
    return { text: protectedText, mathBlocks };
}
```

---

## 四、RAG 引用与溯源标准化

### 4.1 引用格式

LLM 输出中使用 `[1]`, `[2]` 格式引用。来源通过独立的 `sources` SSE 事件传递，结构为：

```json
{
  "type": "sources",
  "data": [
    {"id": 1, "title": "文档标题...", "content": "完整内容..."},
    {"id": 2, "title": "另一文档", "content": "..."}
  ]
}
```

### 4.2 引用渲染实现流程

```
1. retriever 检索完成
   → 后端发射 sources 事件（含唯一 id、title、content）
   → 前端将 docs 累积到 allSources 数组

2. 实时 token 渲染期间（每个 is_final=true token 后）
   → formatCitationHtml(): [1] → <span class="citation-link" onclick="scrollToReference(1, msgIdx)">[1]</span>
   → 引用角标可点击，点击后滚动至底部引用卡片

3. final 事件触发时
   → renumberCitations(): 将绝对 doc ID 重映射为出现顺序的序号 [1][2][3]
   → renderSources(): 在消息底部渲染可展开的引用卡片（含标题和内容）
```

**关键代码位置**：

| 函数 | 文件 | 行号 | 作用 |
|------|------|------|------|
| `formatCitationHtml()` | `main.js` | 3483 | `[N]` → 可点击 `<span>` |
| `renumberCitations()` | `main.js` | 3449 | 重映射绝对 ID 为序号 |
| `renderSources()` | `main.js` | 4020 | 渲染底部引用卡片列表 |
| `scrollToReference()` | `main.js` | ~4000 | 点击角标跳转到引用卡片 |

### 4.3 是否能实时引用？

**部分实时**：

| 内容 | 实时性 | 说明 |
|------|--------|------|
| `[N]` 角标超链接 | ✅ 实时 | 随每个 `is_final=true` token 追加并渲染 |
| 引用卡片（来源详情） | ❌ 非实时 | 仅在 `final` 事件触发后渲染 |
| Hover Tooltip (Gemini 风格) | ❌ 未实现 | 只有点击跳转，无悬停预览 |

**与 EST 的对比**：EST "生成完才渲染"的问题，UltraRAG 已在 token 级别实现实时 Markdown 渲染；`[N]` 角标随文字实时出现且可点击。但 Gemini 风格的悬停卡片未实现。

---

## 五、其他功能

### 5.1 工作流可视化编排

**有，但非图形化拖拽。**

- Pipeline Builder 页面：通过 Node Picker 弹窗向 pipeline 列表中插入节点（tool / branch / loop 三种类型）
- Pipeline 以 **YAML 列表**存储，前端以列表形式渲染，非 DAG 拓扑图
- **无拖拽连线等可视化图形编排**（vs. Dify、n8n）
- 实现位置：`main.js:5406+`，`nodePickerState`，`nodePickerModal`

### 5.2 Subagent 及其实现

**无专用 subagent 架构，但通过 loop + branch 模拟迭代 agent。**

- `loop` + `branch` + `router` 节点组合模拟多轮推理-检索循环
- `router` 服务器提供状态路由判断逻辑，返回 `{data, state}` 格式的包装列表
- `UltraData` 类通过 wrapped list 格式（`{data, branch1_state, ...}`）实现批处理场景下的分支感知数据流
- 支持同一 batch 内的不同 item 走不同分支（adaptive RAG 模式）
- **无并行多 agent 协调，无跨 pipeline 通信**
- 典型案例：`examples/AgentCPM-Report.yaml`（140 轮循环，4+ 状态分支）

### 5.3 其他功能概览

| 功能 | 支持情况 | 说明 |
|------|---------|------|
| 知识库管理（索引/分块/检索） | ✅ 完整 | 支持 BM25、向量、Milvus |
| 多轮对话 | ✅ | 首轮走完整 pipeline，后续轮次走 `LocalGenerationService` 直接生成 |
| 后台异步任务 | ✅ | `/background` 端点，202 Accepted |
| 会话历史持久化 | ✅ | 本地 JSON，含中断恢复 |
| 导出（Markdown / DOCX） | ✅ | 含引用重编号 |
| 国际化 (zh/en) | ✅ | `i18n/zh.js` + `i18n/en.js` |
| 联网搜索 | 🟡 部分 | 有 `ragflow_retriever` 服务器，非原生 UI 集成 |
| Memory Snapshot | ✅ | 每步执行后记录 `global_vars` 快照写入 JSON，用于调试/回放 |

---

## 六、框架整体评价

### 优势与可延展性

1. **MCP 架构高度模块化**：每个服务（retriever、generation、prompt、router 等）独立为 MCP Server，可插拔替换。新增检索源或生成后端只需添加新 server，无需修改核心代码。
2. **流式协议清晰**：SSE + 结构化 JSON 事件类型，`step_start/step_end` 可直接用于进度条渲染，`depth` 字段支持嵌套层级。
3. **Markdown 栈完整**：marked + KaTeX + hljs + DOMPurify 组合健全，且有 LaTeX 数学公式保护机制，实测对常见渲染场景覆盖完整。
4. **实时渲染**：每个 `is_final=true` token 触发完整 Markdown 重渲染，用户感知流畅。
5. **Pipeline YAML 直观**：声明式 pipeline 配置，loop/branch/tool 结构清晰，便于快速搭建新 RAG 流程。
6. **Apache 2.0**：可直接商业使用和修改，只需保留许可声明。

### 缺点

1. **无拓扑图进度渲染能力**：`step_start/step_end` 传递的是 `{name, depth}`，没有节点状态枚举（进行中/完成/失败），前端无法直接渲染 DAG 拓扑图高亮，只能做线性进度条。
2. **引用 Hover Tooltip 缺失**：只有 `[N]` 点击跳转，无 Gemini 风格的悬停预览卡片；引用卡片仅在 `final` 后渲染，非完全实时。
3. **前端为 10k+ 行 Vanilla JS 单文件**：`main.js` 无组件化，耦合度高，难以按需扩展或进行单元测试。
4. **Pipeline Builder 非图形化**：仅列表 + 弹窗方式添加节点，无拖拽可视化编排，对复杂 pipeline 的可读性较差。
5. **无真正 subagent**：不支持并行子任务，loop 节点迭代控制依赖外部路由器，调试相对复杂。
6. **`is_final` flag 设计较非标准**：将"思考 token"与"最终 token"混在同一 `token` 事件中靠布尔字段区分，语义不够清晰。

### 是否可直接使用

**✅ 可以**，开源协议为 **Apache 2.0**（版权归 OpenBMB），允许商业使用、修改和分发，只需保留许可声明。

---

## 七、关键代码位置速查表

| 关注点 | 文件 | 行号 |
|--------|------|------|
| SSE 生成器函数 | `ui/backend/pipeline_manager.py` | 920–1010 |
| 事件 `step_start` 发射 | `src/ultrarag/client.py` | 1469–1471 |
| 事件 `sources` 发射 | `src/ultrarag/client.py` | 1698–1700 |
| 事件 `token` (LLM streaming) | `src/ultrarag/client.py` | 1757–1764 |
| Loop 终止机制 | `src/ultrarag/client.py` | 236–240, 1474–1495 |
| Branch/Router 执行 | `src/ultrarag/client.py` | 1496–1551 |
| UltraData 全局状态管理 | `src/ultrarag/client.py` | 309–954 |
| Memory Snapshot 写入 | `src/ultrarag/client.py` | 889–944 |
| 前端 SSE 流解析主循环 | `ui/frontend/main.js` | 4492–4626 |
| `step_start/step_end` → 进度 UI | `ui/frontend/main.js` | 4204–4360 |
| `renderMarkdown()` | `ui/frontend/main.js` | 2481–2521 |
| `protectMath()` / `restoreMath()` | `ui/frontend/main.js` | 1980–2000 |
| `renderLatex()` | `ui/frontend/main.js` | 2002–2021 |
| `renderCodeBlock()` + copy | `ui/frontend/main.js` | 2024–2057 |
| `applyTableEnhancements()` | `ui/frontend/main.js` | 2074–2120 |
| `formatCitationHtml()` | `ui/frontend/main.js` | 3483–3491 |
| `renumberCitations()` | `ui/frontend/main.js` | 3449–3481 |
| `renderSources()` | `ui/frontend/main.js` | 4020–4070 |
| Pipeline YAML 示例（vanilla） | `examples/vanilla_rag.yaml` | 全文 |
| Pipeline YAML 示例（agentic） | `examples/search_r1.yaml` | 全文 |
| Pipeline YAML 示例（复杂 agent）| `examples/AgentCPM-Report.yaml` | 全文 |

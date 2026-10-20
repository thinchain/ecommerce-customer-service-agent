# 前端 UI 结构与扩展说明

## 1. 原页面的设计

原来的 app/page.tsx 是一个客户端组件，`"use client"` 允许使用 React 状态、事件和浏览器 API。页面由居中的卡片、标题、问题表单、回复区域组成。

- `message` 存放输入框内容；textarea 的 value 与 onChange 组成受控输入。
- `reply` 存放唯一一条回复；再次发送时 setReply("") 清掉旧回复。
- `error` 存放错误说明；`isLoading` 控制按钮禁用与加载提示。
- handleSubmit 阻止表单默认刷新，发送 POST /api/chat，解析 JSON 并更新状态。
- JSX 根据 loading/error/reply 决定显示什么。状态更新后 React 重新计算 UI，无须手工修改 DOM。
- globals.css 决定卡片宽度、颜色、间距、圆角、字体和移动端布局。

因此原页面是一次问答表单，只有一个回复位置，没有历史消息的数据结构。

## 2. 改造后的文件分工

| 文件 | 职责 |
| --- | --- |
| frontend/app/layout.tsx | 页面根布局、全局 CSS 引入、中文语言标记、标题与描述元数据 |
| frontend/app/page.tsx | 页面入口、状态持有者、发送请求、清空记录、各组件组装 |
| frontend/components/chat-message.tsx | Message 类型及单条消息渲染，区分用户/助手和等待/成功/失败 |
| frontend/components/chat-composer.tsx | 受控输入框、字数提示、快捷键、发送按钮、输入框自动增高 |
| frontend/app/globals.css | 布局、颜色、消息外观、交互反馈、响应式适配 |
| frontend/next.config.ts | 把同源 /api/* 请求转发给 FastAPI，默认 127.0.0.1:8000 |
| frontend/package.json | 依赖声明和 dev/build/start 命令 |
| frontend/package-lock.json | 安装依赖时锁定具体版本，通常由 npm 维护 |
| frontend/tsconfig.json | TypeScript 类型检查、JSX 和 @/* 路径映射 |
| frontend/next-env.d.ts | Next.js 生成的类型引用，通常不手工修改 |

components 文件由客户端 page 导入，属于客户端组件树。它们接收 props，不自行调用后端。这样展示样式与请求逻辑可以分别调整。

## 3. 页面各块

- sidebar：展示产品名、清空按钮、第一条问题作为当前对话标题及提问数。这里只概览当前对话，没有多个历史会话的切换功能。
- chat-header：显示 Qwen、本地助手与清空入口。取消原先写死的 GPU4 文案；前端标题不决定实际显卡。
- welcome：没有消息时出现欢迎语与示例问题。点击示例只填充输入框，用户仍可修改再发送。
- transcript：独立滚动区域。message-list 遍历消息数组，用稳定 id 作为 React key。
- ChatMessage：用户提问右对齐；助手回复左对齐；等待显示“正在思考”；失败在本条记录里显示错误。
- composer-area：位于主区域底部，不随消息列表滚走。textarea 保留换行，最多 4000 个输入字符单位；Enter 发送，Shift+Enter 换行；中文输入法组合期间不触发 Enter 发送。
- session-note：说明本机持久化、近期上下文及清空行为。

CSS 使用 Flex 将界面撑满视口，100dvh 适配移动浏览器可见高度；transcript 的 min-height: 0 允许它在剩余空间里滚动。内容最大宽度 840px，限制长行。900px 以下示例卡片纵向排列，640px 以下隐藏侧栏，顶部仍可清空。focus-visible、输入标签和 role=log 提供键盘/辅助技术支持；减少动态效果偏好会关闭思考动画。

## 4. 状态与一次发送的完整过程

```text
输入文字 → onChange → page 的 draft 状态 → 输入框重新渲染
提交表单 / Enter → sendMessage
  → 校验非空、长度及是否有请求进行中
  → messages 追加 user 消息 + pending assistant 消息
  → 清空 draft，显示等待状态
  → fetch POST /api/chat，body 为 {message: 当前问题, session_id: 会话 UUID}
  → Next rewrite → FastAPI Agent → vLLM / 工具循环 → 返回 reply、tool_calls 和 memories
  → 按 answerId 替换对应助手消息的 content/status
  → React 更新列表，滚动到最新消息
```

`messages` 是 Message[]，每项有 id、role、content、status。`draft` 与历史消息分开，发送后可继续起草下一条。`isLoading` 控制 UI；pending ref 同步阻止重复请求，避免状态更新尚未渲染时重复发送。controller ref 用于组件卸载时中止浏览器 fetch（不保证模型服务立即停止计算）。input/transcript ref 访问输入框与滚动容器。

请求失败保留用户问题及错误记录；可以复制原问题重新发送。等待时禁用清空，避免回复落到已清空的列表。成功响应校验 reply 为非空字符串；非 JSON 错误响应显示 HTTP 状态对应的通用错误。

会话 ID 保存在 localStorage，成功问答和记忆保存在后端 SQLite；刷新时 GET /api/sessions/{id} 恢复，恢复失败时阻止发送以免误入新会话。首次发送或保存记忆时创建会话。模型参考近期完整问答对和显式长期记忆，按预算限制上下文。失败的问答只在当前页面显示，不写入数据库。

顶部“记忆”按钮打开管理面板，支持新增、同名更新和删除。清空按钮会 DELETE 整个会话，连同历史和记忆一起删除。工具执行记录附在助手消息的折叠区域，显示工具名称、轮数、参数、结果及成功/失败状态。所有记录通过 React 文本节点渲染，不执行 HTML。本版本仍在整轮完成后一次性显示结果，没有流式输出。

## 5. 如何调整或新增功能

### 只调整外观

颜色修改 globals.css 的 :root 变量；侧栏宽度改 .sidebar；对话宽度改 .message-list 和 .composer-area；气泡改 .message-user .message-content；移动端规则改 @media。标题、示例问题改 page.tsx；浏览器标签标题改 layout.tsx metadata。

### 增加纯前端交互

例如复制回复：在 ChatMessage 内增加按钮，通过 navigator.clipboard.writeText(message.content) 复制，并增加成功/失败反馈状态。收藏、折叠长回复也可在展示组件或页面状态中实现；是否跨刷新保留需单独设计存储。

### 增加消息类型

当要显示图片、文件或工具执行结果，应将单一 content 扩展成带 kind 的联合类型，例如：

```ts
type MessagePart =
  | { kind: "text"; text: string }
  | { kind: "image"; url: string; alt: string }
  | { kind: "tool-result"; toolName: string; summary: string };
```

然后新增 TextPart/ImagePart/ToolResultPart 组件，让 ChatMessage 根据 kind 渲染。若来自后端，必须同时修改响应 schema 和响应解析；只改变 TypeScript 类型不会让服务器产生这些数据。图片上传还需要文件选择器、上传接口、文件大小/类型校验，以及支持视觉输入的模型。

### 增加独立业务

例如文档问答：新增 app/documents/page.tsx 作为 /documents 页面、对应 components/documents/* 展示组件和上传/问答状态；后端增加独立 documents router、schema、service，并在 main 中 include_router。现有 /api/* rewrite 通常可以继续转发。共享组件可复用，不应把所有业务塞进聊天消息组件。

### 多会话、上下文与 Agent

- 多会话切换：新增 conversations 数组、activeConversationId；每个会话拥有自己的 messages。侧栏改成会话按钮列表，明确删除与新建行为。
- 上下文：后端已按预算构建近期问答和长期记忆；进一步可增加 tokenizer 精确预算或历史摘要。系统提示词由后端控制。
- 流式输出：前后端增加流式协议，客户端逐块更新同一 assistant 消息，还要处理断流、取消和半条回复。
- ReAct/Agent：后端已实现工具白名单、参数校验、循环与超时限制，前端展示整轮工具记录。进一步可增加 SSE，实时展示执行进度。

## 6. 内容与边界

本次消息通过 React 文本节点渲染，不解释回复中的 HTML。以后增加 Markdown 时，应禁用原始 HTML或使用经过审查的净化流程；不要直接使用 dangerouslySetInnerHTML 注入模型输出。API 密钥保留在后端，不能放进页面或 NEXT_PUBLIC_*。前端长度限制用于交互体验，后端仍必须校验输入；工具执行权限也必须在后端校验。

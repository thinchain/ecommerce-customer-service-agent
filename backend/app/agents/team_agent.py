"""Commerce agent backed by the existing LangGraph orchestration."""
import asyncio
from collections.abc import Callable

from langgraph.errors import GraphRecursionError, NodeCancelledError

from app.agents.context import build_context
from app.agents.graph import build_team_graph
from app.agents.policy import customer_gate, danger_reason
from app.agents.recall import answer_history
from app.agents.roles import SUPERVISOR_PROMPT
from app.agents.business_reply import compose_business_reply
from app.core.agent import Agent, AgentResult, RunContext
from app.core.config import AgentConfig, Settings, get_settings, override_settings
from app.core.exceptions import ModelServiceError
from app.core.llm import LLM


class LangGraphTeamAgent(Agent):
    def __init__(self, name: str, llm: LLM, executor: Callable, *,
                 settings: Settings | None = None, config: AgentConfig | None = None,
                 system_prompt: str = SUPERVISOR_PROMPT):
        self.settings = settings or get_settings()
        super().__init__(name, llm, config=config or self.settings.agent, system_prompt=system_prompt)
        self.executor = executor
        self.graph = build_team_graph(
            llm, executor, max_steps=self.config.max_steps, worker_steps=self.config.worker_steps,
            max_handoffs=self.config.max_handoffs, max_tool_calls=self.config.max_tool_calls,
            context_bytes=self.config.context_bytes, report_bytes=self.config.handoff_report_bytes,
            max_calls=self.config.max_calls_per_response, settings=self.settings,
        )

    async def run(self, input_text: str, *, context: RunContext) -> AgentResult:
        try:
            # Admission classification and history-only execution share the same deadline.
            with override_settings(self.settings):
                async with asyncio.timeout(self.config.timeout_seconds):
                    return await self._execute(input_text, context)
        except TimeoutError as exc:
            raise ModelServiceError(504, 'Agent 执行超时，请缩小问题范围后重试。') from exc
        except GraphRecursionError as exc:
            raise ModelServiceError(502, '多 Agent 图达到执行上限，请缩小任务后重试。') from exc
        except NodeCancelledError as exc:
            raise asyncio.CancelledError() from exc

    async def _execute(self, question: str, context: RunContext) -> AgentResult:
        history = [message.to_storage() for message in context.history]
        memories = dict(context.memories)
        gate = await customer_gate(question, history, settings=self.settings)
        if gate['status'] == 'blocked':
            return AgentResult(reply=gate['reply'], memories=memories,
                               decision={'scope': gate}, verification={'policy': 'blocked'})
        if gate.get('action') == 'history_only':
            reply, verification = await answer_history(question, history, self.llm, self.config.context_bytes)
            if self.settings.catalog_evidence_enabled:
                trace = next((m.get('tool_calls', []) for m in reversed(history)
                              if m.get('role') == 'assistant' and m.get('tool_calls')), [])
                # Explicit current-value requests must not reuse historical observations.
                import re
                current = bool(re.search(r'现在|当前|重新查|最新|\b(?:current|latest|refresh)\b', question, re.I))
                rendered = compose_business_reply(question, reply, [] if current else trace,
                                                  historical=not current)
                if rendered is not None:
                    reply, parts = rendered
                    verification = {'parts': parts, 'evidence_scope': 'current_unavailable' if current else 'historical_turn',
                                    'semantic_factuality': 'not_measured', 'invalid': []}
                    if 'catalog' in parts:
                        verification['catalog'] = parts['catalog']
            if danger_reason(reply):
                reply = '历史回顾包含不允许提供的操作内容，已阻止。'
                verification['policy'] = 'output_blocked'
            return AgentResult(reply=reply, memories=memories,
                               decision={'status': 'not_requested', 'scope': gate}, verification=verification)
        messages = build_context(history, memories, question, budget=self.config.context_bytes,
                                 system_prompt=self.system_prompt)
        result = await self.graph.ainvoke({
            'base': messages[:2], 'history': messages[2:-1], 'current': messages[-1],
            'rounds': [], 'pending': [], 'steps': 0, 'handoffs': 0,
            'trace': [], 'events': [], 'memories': memories, 'stopped': False,
            # Carry only a uniquely verified entity, never historical numeric values.
            'product_targets': self._recent_product_targets(history),
        }, config={'recursion_limit': (self.config.max_steps + 1) * (self.config.max_calls_per_response + 2) + 5,
                   'metadata': {'run_id': context.run_id, 'session_id': context.session_id}})
        result['decision']['scope'] = gate
        if danger_reason(result['reply']):
            result['reply'] = '回复包含不允许提供的操作内容，已阻止。请提出商品、本人订单或售后问题。'
            result['verification']['policy'] = 'output_blocked'
        return AgentResult(**{key: result[key] for key in AgentResult.model_fields})

    @staticmethod
    def _recent_product_targets(history):
        for message in reversed(history):
            verification = message.get('verification', {})
            catalog = verification.get('catalog')
            if catalog is not None:
                ids = list(dict.fromkeys(f['product_id'] for f in catalog.get('facts', [])))
                return ids if len(ids) == 1 else []
            knowledge = verification.get('parts', {}).get('knowledge')
            if knowledge is not None:
                ids = list(dict.fromkeys(q['product_id'] for q in knowledge.get('quotes', []) if q.get('product_id')))
                return ids if len(ids) == 1 else []
        return []

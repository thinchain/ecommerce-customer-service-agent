"""Application boundary: claim a session, execute an agent, then commit a complete turn."""
from app.agents.context import build_context as _build_context, size
from app.agents.team_agent import LangGraphTeamAgent
from app.core.agent import Agent, AgentResult, RunContext
from app.core.config import Settings, get_settings
from app.core.message import ConversationMessage
from app.services.history import execution_state
from app.services.qwen import complete
from app.services.tools import execute_tool


def build_context(history, memories, question):
    """Compatibility entry point for callers of the previous context builder."""
    return _build_context(history, memories, question, budget=get_settings().agent_context_bytes)


async def run_agent(question, session_id, store, *, agent: Agent | None = None,
                    settings: Settings | None = None):
    settings = settings or get_settings()
    runner = agent or LangGraphTeamAgent('电商客服', complete, execute_tool, settings=settings)
    with store.edit(session_id, timeout_seconds=runner.config.timeout_seconds) as session:
        context = RunContext(session_id=session_id,
                             history=[ConversationMessage.from_storage(item) for item in session['messages']],
                             memories=dict(session['memories']))
        result = await runner.run(question, context=context)
        return save_turn(question, session_id, store, session, result,
                         max_history_turns=runner.config.max_history_turns)


def save_turn(question, session_id, store, session, result: AgentResult, *, max_history_turns=50):
    trace = [item.model_dump() for item in result.trace]
    events = [item.model_dump(exclude_unset=True) for item in result.events]
    history = session['messages'] + [
        ConversationMessage.new('user', question).to_storage(),
        ConversationMessage.new('assistant', result.reply, tool_executions=result.trace,
                                execution_state=execution_state(trace), agent_events=result.events,
                                decision=result.decision, verification=result.verification).to_storage(),
    ]
    store.save(session_id, history[-2 * max_history_turns:], result.memories)
    return {'reply': result.reply, 'session_id': session_id, 'tool_calls': trace,
            'memories': result.memories, 'stopped_by_limit': result.stopped, 'agent_events': events,
            'decision': result.decision, 'verification': result.verification}

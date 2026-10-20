"""Separate raw UI history from model context, without intent/keyword heuristics."""
import json


def execution_state(trace):
    observations = [call for call in trace if not call.get('name', '').startswith('memory_')]
    if not observations:
        return 'no_observation'
    successes = sum(call.get('result', {}).get('ok') is True for call in observations)
    return 'failed' if successes == 0 else 'succeeded' if successes == len(observations) else 'mixed'


def prepare_history(history):
    prepared = []
    for index in range(0, len(history) - 1, 2):
        user, assistant = history[index:index+2]
        if assistant.get('decision', {}).get('scope', {}).get('status') == 'blocked':
            # Keep refusals in the UI but do not replay rejected instructions to Qwen.
            continue
        trace = assistant.get('tool_calls', [])
        agents = [{'agent': event['agent'], 'status': event['status']}
                  for event in assistant.get('agent_events', []) if event.get('agent') != 'supervisor']
        state = execution_state(trace)
        content = assistant['content']
        worker_failure = bool(agents) and all(event['status'] in ('failed', 'limit') for event in agents)
        if state == 'failed' or (state == 'no_observation' and worker_failure):
            # Execution status is not a claim about whether the whole task succeeded.
            content = '该轮工具未取得有效结果。原问题仍可作为上下文；当前工具状态未知，需要重新判断。'
        if trace or agents:
            summary = {'scope': 'historical_turn_only', 'execution_state': state,
                       'agents': agents,
                       'observations': [{'tool': call['name'], 'ok': call.get('result', {}).get('ok')}
                                        for call in trace[:32]]}
            content += '\n历史执行摘要（非当前状态）：' + json.dumps(summary, ensure_ascii=False)
        prepared.extend([user, {**assistant, 'content': content}])
    return prepared

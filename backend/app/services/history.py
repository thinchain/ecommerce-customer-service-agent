"""Execution status reporting and filtering of rejected conversation turns."""


def execution_state(trace):
    observations = [call for call in trace if not call.get('name', '').startswith('memory_')]
    if not observations:
        return 'no_observation'
    successes = sum(call.get('result', {}).get('ok') is True for call in observations)
    return 'failed' if successes == 0 else 'succeeded' if successes == len(observations) else 'mixed'


def prepare_history(history):
    """Keep permitted question/answer pairs without rewriting their content."""
    prepared = []
    for index in range(0, len(history) - 1, 2):
        user, assistant = history[index:index+2]
        if assistant.get('decision', {}).get('scope', {}).get('status') == 'blocked':
            # Keep refusals in the UI but do not replay rejected instructions to Qwen.
            continue
        prepared.extend([user, assistant])
    return prepared

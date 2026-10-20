"""Server-side admission policy; classifiers never grant tool permissions."""
import re
import unicodedata

from app.agents.decisions import scope_advice
from app.core.config import Settings, get_settings


def danger_reason(text):
    text = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
    patterns = {
        'destructive_command': r'rm-{1,2}[a-z]*r[a-z]*f|rm-{1,2}[a-z]*f[a-z]*r|droptable|dropdatabase|formatc:|删除.{0,12}(数据库|全部订单|所有订单|系统文件)',
        'credential_theft': r'(窃取|盗取|获取|导出|泄露).{0,12}(密码|密钥|token|凭据)|steal.{0,12}(password|credential|token)',
        'bypass_policy': r'(忽略|无视|绕过).{0,12}(系统提示|系统指令|权限|限制|安全规则)|ignore.{0,12}(previousinstructions|systemprompt)|bypass.{0,12}(permission|security)',
        'unauthorized_identity': r'(切换|冒充|改成).{0,12}(用户|身份|u\d{5})|(查询|查看|导出|列出).{0,15}(其他用户|别人的|所有用户|全部用户).{0,10}(订单|资料|信息)',
        'attack': r'(编写|生成|执行|运行).{0,12}(恶意软件|勒索软件|木马|攻击脚本|漏洞利用)|执行.{0,8}(shell|系统命令|bash|powershell)',
    }
    return next((reason for reason, pattern in patterns.items() if re.search(pattern, text)), None)


async def customer_gate(question, history, *, settings: Settings | None = None):
    settings = settings or get_settings()
    reason = danger_reason(question)
    if reason:
        return {'status': 'blocked', 'source': 'server_rule', 'reason': reason,
                'reply': '该请求涉及越权访问或危险操作，客服系统不予执行。只能查询当前固定用户的订单。'}
    if not settings.jev_scope_enabled:
        return {'status': 'disabled'}
    if re.fullmatch(r'(你好|您好|谢谢|多谢|再见|hello|hi)[！!。,.，\s]*', question, re.I):
        return {'status': 'allowed', 'source': 'server_rule', 'reason': 'greeting'}
    # Recent permitted user questions help resolve references; never send order
    # observations or replay rejected instructions to the classifier.
    previous = []
    for index in range(0, len(history) - 1, 2):
        user, answer = history[index:index + 2]
        if answer.get('decision', {}).get('scope', {}).get('status') != 'blocked':
            previous.append(user['content'][:300])
    advice = await scope_advice(question, history=previous[-3:], settings=settings)
    gate = {'source': 'kev', 'evaluation': advice}
    if advice.get('status') != 'evaluated':
        return {**gate, 'status': 'blocked', 'reason': 'scope_unavailable',
                'reply': '客服范围校验暂时不可用，请稍后重试。'}
    choice = advice['choice']
    if choice == 'customer_service':
        return {**gate, 'status': 'allowed'}
    if choice == 'conversation_recall':
        if re.search(r'重新查|再查|最新|实时|核验|核实|(?:现在|当前).{0,12}(?:价格|库存|状态)|\b(?:recheck|refresh|latest|current)\b', question, re.I):
            return {**gate, 'status': 'allowed', 'reason': 'explicit_live_query'}
        return {**gate, 'status': 'allowed', 'action': 'history_only'}
    reply = '我仅处理本店商品、订单、物流与售后问题，请提出与客服相关的问题。'
    if choice == 'dangerous':
        reply = '该请求涉及越权访问或危险操作，客服系统不予执行。'
    elif choice == 'unclear':
        reply = '请说明需要咨询的商品、本人订单或售后问题。'
    return {**gate, 'status': 'blocked', 'reason': choice, 'reply': reply}

"""Conservative requirements for explicit product lookups, independent of Kev."""
import re
from app.agents.catalog_evidence import product_ids


def product_requirements(question):
    # Drafting quoted text is not a request to verify the quoted product claims.
    if re.search(r'改写|翻译|润色', question) and not re.search(r'查询|核实|核验', question):
        return []
    product = re.search(r'SYN-[A-Z]+-\d+|商品|产品|型号|耳机|摄像头|键盘|鼠标|这款|那款|它', question, re.I)
    if not product and not product_ids(question):
        return []
    required = []
    from app.agents.request import requested_fields
    fields = requested_fields(question)
    if 'knowledge' in fields or re.search(r'适用场景|参数|自动对焦', question):
        required.append('knowledge_search')
    if {'price', 'stock'}.intersection(fields):
        required.append('catalog_search')
    return required


def missing_product_evidence(question, trace):
    used = {item['name'] for item in trace if item['result'].get('ok')}
    return [name for name in product_requirements(question) if name not in used]

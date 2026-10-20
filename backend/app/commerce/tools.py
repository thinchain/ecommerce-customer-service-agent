from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.commerce.store import get_commerce
from app.knowledge.store import get_knowledge


class Args(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)


class ListOrders(Args):
    limit: int = Field(default=10, ge=1, le=20)


class OrderDetail(Args):
    order_id: str = Field(min_length=1, max_length=80)


class CatalogSearch(Args):
    query: str = Field(default='', max_length=100)
    max_price_cents: int | None = Field(default=None, ge=0)
    in_stock: bool = False
    product_ids: list[str] | None = Field(default=None, max_length=20)
    limit: int = Field(default=10, ge=1, le=20)


class KnowledgeSearch(Args):
    query: str = Field(min_length=2, max_length=300)
    product_id: str = Field(default='', max_length=160)
    kind: Literal['', 'product', 'product_qa', 'policy'] = ''
    limit: int = Field(default=4, ge=1, le=6)


# One source for tool schemas, validation and dispatch (native Agent + MCP).
BUSINESS_TOOLS = {
    'list_orders': (ListOrders, '查询当前演示用户的订单列表；不知道订单号时先调用。',
                    lambda args: get_commerce().orders(**args)),
    'order_detail': (OrderDetail, '查询本人订单详情、商品 ID、支付、物流、退款与工单。金额单位均为人民币分。不能执行退款。',
                     lambda args: get_commerce().order(**args)),
    'catalog_search': (CatalogSearch, '查询商品当前价格和库存。金额单位为分（500元=50000分）。已知商品 ID 时使用 product_ids；指定 product_ids 后忽略 query 文本，价格和库存筛选仍生效。',
                       lambda args: get_commerce().products(**args)),
    'knowledge_search': (KnowledgeSearch, 'RAG 检索商品功能、兼容性、问答和商店政策。尽可能指定 product_id；政策指定 kind=policy。引用返回的 citation 与来源；不能据此推断实时库存或订单状态。',
                         lambda args: get_knowledge().search(**args)),
}

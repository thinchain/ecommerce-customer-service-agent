"use client";

import { useEffect, useState } from "react";

type Overview = {
  counts?: {products: number; orders: number};
  user_id: string;
  products: { product_id: string; title: string; price_cents: number; stock: number }[];
  orders: { order_id: string; status: string; total_cents: number; created_at: string }[];
};
const statusLabels: Record<string, string> = {
  in_transit: "运输中", delivered: "已签收", delayed: "物流延迟", payment_failed: "支付失败",
  cancelled: "已取消", refund_pending: "退款处理中", refunded: "已退款", partial_refund: "部分退款",
  return_expired: "已签收 · 超退货期", exception: "物流异常",
};

export function CommercePanel({ onSelect, disabled }: { onSelect: (question: string) => void; disabled: boolean }) {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    void fetch("/api/commerce/overview", { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error("暂时无法加载商店，请稍后刷新。");
      setData(await response.json() as Overview);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setError(error instanceof Error ? error.message : "加载失败");
    });
    return () => controller.abort();
  }, []);
  return <section className="commerce-panel" aria-label="演示商店">
    <div className="commerce-heading"><strong>客服业务沙盒</strong><span>{data?.counts ? `${data.counts.products} 个商品 · ${data.counts.orders} 个合成订单` : "加载数据中"} · {data?.user_id}</span></div>
    {error && <p role="alert">{error}</p>}
    {data && <>
      <div className="product-grid">{data.products.map((product) => <button key={product.product_id} disabled={disabled}
        onClick={() => onSelect(`请介绍 ${product.title}（${product.product_id}）的适用场景，并查询价格和库存。`)}>
        <strong>{product.title}</strong><span>¥{(product.price_cents / 100).toFixed(2)}</span>
        <small>{product.stock > 0 ? `库存 ${product.stock}` : "暂时缺货"}</small>
      </button>)}</div>
      <details className="order-overview"><summary>我的演示订单（{data.orders.length}）</summary>
        {data.orders.map((order) => <button key={order.order_id} disabled={disabled}
          onClick={() => onSelect(`查询订单 ${order.order_id} 的商品、物流和退款状态。`)}>
          <span>{order.order_id}</span><span>{statusLabels[order.status] || order.status}</span>
          <span>¥{(order.total_cents / 100).toFixed(2)}</span>
        </button>)}
      </details>
      {!data.products.length && <p>尚未导入演示商品，请按项目 README 初始化数据。</p>}
    </>}
  </section>;
}

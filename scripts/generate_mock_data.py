"""Idempotent synthetic data generator; existing orders/products are never overwritten."""
import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.commerce.store import CommerceStore
from app.core.config import get_settings

PRODUCTS = [
    ('DEMO-RUN', '轻跑运动耳机', 'Demo Audio', 29900, 18),
    ('DEMO-DESK', '桌面降噪耳机', 'Demo Audio', 49900, 8),
    ('DEMO-BUDS', '迷你无线耳机', 'Demo Audio', 19900, 0),
    ('DEMO-KEY', '便携蓝牙键盘', 'Demo Office', 15900, 23),
]
SCENARIOS = ['in_transit', 'delivered', 'delayed', 'payment_failed', 'cancelled',
             'refund_pending', 'refunded', 'partial_refund', 'return_expired', 'exception']


def generate(path=None, users=10, orders=40, seed=42, as_of=date(2026, 10, 2)):
    path = path if path is not None else get_settings().commerce_db_path
    rng = random.Random(seed)
    store = CommerceStore(path)
    with store.connection() as db:
        for pid, title, brand, price, stock in PRODUCTS:
            db.execute('INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?)',
                       (pid, title, brand, price, stock, 'synthetic_demo', pid, 'demo'))
        for index in range(1, users + 1):
            uid = f'U{index:05d}'
            db.execute('INSERT OR IGNORE INTO users VALUES (?,?,?)', (uid, f'演示用户{index}', f'demo{index}@example.invalid'))
        catalog = db.execute('SELECT product_id,price_cents FROM products ORDER BY product_id').fetchall()
        for index in range(orders):
            oid = f'ORD{index+1:05d}'
            if db.execute('SELECT 1 FROM orders WHERE order_id=?', (oid,)).fetchone():
                continue
            status = SCENARIOS[index % len(SCENARIOS)]
            # The first 10 scenarios belong to the trusted demonstration customer.
            uid = 'U00001' if index < 10 else f'U{rng.randint(1, users):05d}'
            product = catalog[rng.randrange(len(catalog))]
            if index == 0:
                product = ('DEMO-RUN', 29900)
            quantity = rng.randint(1, 2)
            total = product[1] * quantity
            created = as_of - timedelta(days=45 if status == 'return_expired' else 7)
            db.execute('INSERT INTO orders VALUES (?,?,?,?,?)', (oid, uid, status, total, created.isoformat()))
            db.execute('INSERT INTO order_items VALUES (?,?,?,?)', (oid, product[0], quantity, product[1]))
            paid = status not in ('payment_failed', 'cancelled')
            db.execute('INSERT INTO payments VALUES (?,?,?,?,?)',
                       (f'PAY{index+1:05d}', oid, total, 'paid' if paid else 'failed' if status == 'payment_failed' else 'void', 'demo_wallet'))
            if paid:
                delivered = status in ('delivered', 'refund_pending', 'refunded', 'partial_refund', 'return_expired')
                delivery = created + timedelta(days=3)
                db.execute('INSERT INTO shipments VALUES (?,?,?,?,?,?,?)', (
                    f'SHP{index+1:05d}', oid, '演示快递', f'DEMO{index+1:08d}',
                    'delivered' if delivered else status, (created+timedelta(days=5)).isoformat(),
                    delivery.isoformat() if delivered else None))
            if status in ('refund_pending', 'refunded', 'partial_refund'):
                db.execute('INSERT INTO refunds VALUES (?,?,?,?,?)', (
                    f'REF{index+1:05d}', oid, total // 2 if status == 'partial_refund' else total,
                    'pending' if status == 'refund_pending' else 'succeeded', '演示售后申请'))
            if status in ('delayed', 'exception'):
                db.execute('INSERT INTO tickets VALUES (?,?,?,?)', (f'TKT{index+1:05d}', oid, 'shipping', 'open'))
    return {'users_requested': users, 'orders_requested': orders, 'as_of': as_of.isoformat()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=get_settings().commerce_db_path)
    parser.add_argument('--users', type=int, default=10)
    parser.add_argument('--orders', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--as-of', type=date.fromisoformat, default=date(2026, 10, 2))
    args = parser.parse_args()
    if not 1 <= args.users <= 100000 or not 1 <= args.orders <= 500000:
        parser.error('users 必须为 1..100000，orders 必须为 1..500000')
    print(json.dumps(generate(args.db, args.users, args.orders, args.seed, args.as_of), ensure_ascii=False))

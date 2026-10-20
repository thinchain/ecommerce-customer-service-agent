"""Parameterized business queries. Model callers never submit SQL or customer IDs."""
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

SCHEMA = '''
CREATE TABLE IF NOT EXISTS users(user_id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS products(
 product_id TEXT PRIMARY KEY, title TEXT NOT NULL, brand TEXT NOT NULL DEFAULT '',
 price_cents INTEGER NOT NULL CHECK(price_cents>=0), stock INTEGER NOT NULL CHECK(stock>=0),
 source TEXT NOT NULL, source_id TEXT NOT NULL, locale TEXT NOT NULL DEFAULT 'demo');
CREATE TABLE IF NOT EXISTS orders(
 order_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id),
 status TEXT NOT NULL, total_cents INTEGER NOT NULL CHECK(total_cents>=0), created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS orders_owner ON orders(user_id, created_at);
CREATE TABLE IF NOT EXISTS order_items(
 order_id TEXT NOT NULL REFERENCES orders(order_id), product_id TEXT NOT NULL REFERENCES products(product_id),
 quantity INTEGER NOT NULL CHECK(quantity>0), unit_price_cents INTEGER NOT NULL CHECK(unit_price_cents>=0),
 PRIMARY KEY(order_id,product_id));
CREATE TABLE IF NOT EXISTS payments(
 payment_id TEXT PRIMARY KEY, order_id TEXT UNIQUE NOT NULL REFERENCES orders(order_id),
 amount_cents INTEGER NOT NULL CHECK(amount_cents>=0), status TEXT NOT NULL, method TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shipments(
 shipment_id TEXT PRIMARY KEY, order_id TEXT UNIQUE NOT NULL REFERENCES orders(order_id),
 carrier TEXT NOT NULL, tracking_no TEXT NOT NULL, status TEXT NOT NULL,
 expected_delivery TEXT, delivered_at TEXT);
CREATE TABLE IF NOT EXISTS refunds(
 refund_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id),
 amount_cents INTEGER NOT NULL CHECK(amount_cents>0), status TEXT NOT NULL, reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tickets(
 ticket_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id),
 issue_type TEXT NOT NULL, status TEXT NOT NULL);
'''


class CommerceStore:
    def __init__(self, path=None, *, user_id=None):
        settings = get_settings()
        self.path = Path(path if path is not None else settings.commerce_db_path)
        self.user_id = user_id if user_id is not None else settings.demo_user_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def products(self, query='', max_price_cents=None, in_stock=False, product_ids=None, limit=10):
        clauses, args = ['1=1'], []
        # Explicit IDs identify candidates; a model-supplied display name must not
        # accidentally exclude them. Price and stock filters still apply.
        if query and product_ids is None:
            # Treat %, _ and backslash literally, rather than as SQL LIKE operators.
            escaped = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            clauses.append("(title LIKE ? ESCAPE '\\' OR product_id=?)")
            args.extend([f'%{escaped}%', query])
        if max_price_cents is not None:
            clauses.append('price_cents<=?')
            args.append(max_price_cents)
        if in_stock:
            clauses.append('stock>0')
        if product_ids is not None:
            if not product_ids:
                return []
            clauses.append('product_id IN (' + ','.join('?' for _ in product_ids) + ')')
            args.extend(product_ids)
        args.append(min(max(limit, 1), 20))
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM products WHERE ' + ' AND '.join(clauses) + ' ORDER BY product_id LIMIT ?', args)]

    def orders(self, user_id=None, limit=10):
        user_id = self.user_id if user_id is None else user_id
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM orders WHERE user_id=? ORDER BY created_at DESC,order_id LIMIT ?',
                (user_id, min(max(limit, 1), 20)))]

    def counts(self):
        with self.connection() as db:
            return {table: db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                    for table in ('products','users','orders','order_items')}

    def order(self, order_id, user_id=None):
        user_id = self.user_id if user_id is None else user_id
        with self.connection() as db:
            order = db.execute('SELECT * FROM orders WHERE order_id=? AND user_id=?', (order_id, user_id)).fetchone()
            if order is None:
                # Same error for a missing order and another customer's order.
                raise ValueError('订单不存在或不属于当前演示用户')
            result = dict(order)
            result['items'] = [dict(row) for row in db.execute('''
                SELECT i.*,p.title FROM order_items i JOIN products p USING(product_id) WHERE order_id=?''', (order_id,))]
            for table in ('payments', 'shipments', 'refunds', 'tickets'):
                result[table] = [dict(row) for row in db.execute(f'SELECT * FROM {table} WHERE order_id=?', (order_id,))]
            result['data_origin'] = 'synthetic_demo'
            return result


def get_commerce():
    settings = get_settings()
    return _get_commerce(settings.commerce_db_path, settings.demo_user_id)


@lru_cache
def _get_commerce(path, user_id):
    return CommerceStore(path, user_id=user_id)

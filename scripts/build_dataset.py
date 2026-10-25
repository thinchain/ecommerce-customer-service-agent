"""Build a versioned business/knowledge benchmark in a NEW directory, atomically."""
import argparse
import hashlib
import json
import random
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

from app.commerce.store import CommerceStore
from app.core.config import get_settings
from app.knowledge.store import Document, KnowledgeStore
from scripts.generate_mock_data import PRODUCTS, SCENARIOS
from scripts.ingest_knowledge import ingest_file
from scripts.datasets.import_public import convert

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    ('EAR', '运动耳机', '蓝牙', 'Windows、Android、iPhone', '跑步与通勤', '游泳', '双设备连接'),
    ('KEY', '办公键盘', '蓝牙和 USB', 'Windows、macOS', '办公打字', '水下使用', '多设备切换'),
    ('MOU', '无线鼠标', 'USB 接收器', 'Windows、Linux', '移动办公', '蓝牙直连', '静音按键'),
    ('SPK', '便携音箱', '蓝牙', 'Android、iPhone', '户外聚会', '潜水', '立体声配对'),
    ('HUB', '扩展坞', 'USB-C', 'Windows、macOS', '笔记本扩展接口', '给台式机供电', 'HDMI 视频输出'),
    ('MON', '显示器', 'HDMI', '具有 HDMI 输出的电脑', '桌面办公', '触控操作', '高刷新率'),
    ('CAM', '网络摄像头', 'USB', 'Windows、macOS、Linux', '视频会议', '独立录像存储', '自动对焦'),
    ('SSD', '移动固态硬盘', 'USB-C', 'Windows、macOS、Linux', '资料备份', '无线传输', '硬件加密'),
]


def build(output, products=320, users=1000, orders=10000, seed=42, as_of=date(2026, 10, 3), public=None):
    output = Path(output)
    settings = {'version': 1, 'products_generated': products, 'users': users, 'orders': orders,
                'seed': seed, 'as_of': as_of.isoformat(),
                'public_sha256': hashlib.sha256(Path(public).read_bytes()).hexdigest() if public else None}
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['settings'] != settings:
            raise ValueError('目标已有不同版本的数据，请换目录；不覆盖已有业务数据库')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=output.name + '-building-', dir=output.parent))
    try:
        store = CommerceStore(work / 'commerce.sqlite3')
        kb = KnowledgeStore(work / 'knowledge.sqlite3', backend='fts')
        docs, cases = [], []
        with store.connection() as db:
            for pid, title, brand, price, stock in PRODUCTS:
                db.execute('INSERT INTO products VALUES (?,?,?,?,?,?,?,?)', (pid,title,brand,price,stock,'synthetic_demo',pid,'demo'))
            for i in range(products):
                code, category, connection, systems, purpose, unsupported, feature = CATEGORIES[i % len(CATEGORIES)]
                pid = f'SYN-{code}-{i+1:04d}'
                title = f'样本品牌{i%7+1} {category} {i+1:04d}型'
                enabled = (i // len(CATEGORIES)) % 2 == 0
                hours = (8, 12, 24, 36)[(i // 8) % 4]
                price = 5900 + (i * 7919) % 120000
                db.execute('INSERT INTO products VALUES (?,?,?,?,?,?,?,?)',
                           (pid,title,f'合成品牌{i%7+1}',price,0 if i%11==0 else 1+(i*13)%80,'synthetic_benchmark',pid,'demo'))
                specs = f'{title}。接口为{connection}，兼容{systems}，适合{purpose}。不支持{unsupported}。{feature}：{"支持" if enabled else "不支持"}。质保为12个月。'
                if code in ('EAR','KEY','MOU','SPK'):
                    specs += f'标称续航{hours}小时，实际续航随使用环境变化。'
                for suffix, kind, body in [('spec','product',specs),
                    ('qa','product_qa',f'问：{title}是否适合{purpose}？答：可以，需使用{connection}接口；支持的系统为{systems}。'),
                    ('limits','product_qa',f'问：{title}具备{feature}吗？答：{"支持" if enabled else "不支持"}。不支持{unsupported}。')]:
                    docs.append(Document(document_id=f'{pid}:{suffix}',product_id=pid,kind=kind,title=title,
                                         text=body,source='synthetic_benchmark',source_url=f'demo://products/{pid}',
                                         license='Project-generated synthetic fixture',synthetic=True))
                split = 'test' if int(hashlib.sha256(pid.encode()).hexdigest()[:4],16)%5==0 else 'dev'
                cases.append({'id': f'rag-{pid}', 'split':split, 'query':f'{title}连接方式和适用系统是什么',
                              'product_id':pid, 'expected_document_ids':[f'{pid}:spec',f'{pid}:qa'], 'answerable':True})
                cases.append({'id': f'limit-{pid}', 'split':split, 'query':f'请确认{feature}的支持情况',
                              'product_id':pid,'expected_document_ids':[f'{pid}:spec',f'{pid}:limits'],'answerable':True})
            for i in range(1, users+1):
                db.execute('INSERT INTO users VALUES (?,?,?)',(f'U{i:05d}',f'合成用户{i}',f'user{i}@example.invalid'))
        if public:
            convert(Path(public), work/'public-knowledge.jsonl', 'pqa', store, limit=100000)
            ingest_file(work/'public-knowledge.jsonl', kb)
        # Knowledge is independent of dynamic business state.
        ingest_file(ROOT/'fixtures/knowledge/demo.jsonl', kb)
        for doc in docs:
            kb.ingest(doc)
        (work/'synthetic-knowledge.jsonl').write_text('\n'.join(doc.model_dump_json() for doc in docs)+'\n')
        rng = random.Random(seed)
        with store.connection() as db:
            catalog = db.execute('SELECT product_id,price_cents FROM products ORDER BY product_id').fetchall()
            for i in range(orders):
                oid = f'ORD{i+1:05d}'
                state = SCENARIOS[i%len(SCENARIOS)]
                uid = 'U00001' if i<10 or users==1 else f'U{rng.randint(2,users):05d}'
                age = 40+rng.randint(0,15) if state=='return_expired' else rng.randint(1,6) if state=='in_transit' else rng.randint(8,25)
                created = as_of-timedelta(days=age)
                items = rng.sample(catalog, min(rng.randint(1,3),len(catalog)))
                if i==0:items=[('DEMO-RUN',29900)]
                lines=[(p[0],rng.randint(1,3),p[1]) for p in items]
                total=sum(q*p for _,q,p in lines)
                db.execute('INSERT INTO orders VALUES (?,?,?,?,?)',(oid,uid,state,total,created.isoformat()))
                db.executemany('INSERT INTO order_items VALUES (?,?,?,?)',[(oid,p,q,c) for p,q,c in lines])
                paid=state not in ('payment_failed','cancelled')
                db.execute('INSERT INTO payments VALUES (?,?,?,?,?)',(f'PAY{i+1:05d}',oid,total,'paid' if paid else 'failed' if state=='payment_failed' else 'void','synthetic_wallet'))
                if paid:
                    delivered=state in ('delivered','refund_pending','refunded','partial_refund','return_expired')
                    delivery=(created+timedelta(days=3)).isoformat() if delivered else None
                    expected=(created+timedelta(days=7 if state=='in_transit' else 5)).isoformat()
                    db.execute('INSERT INTO shipments VALUES (?,?,?,?,?,?,?)',(f'SHP{i+1:05d}',oid,'合成快递',f'SYN{i+1:010d}','delivered' if delivered else state,expected,delivery))
                if state in ('refund_pending','refunded','partial_refund'):
                    db.execute('INSERT INTO refunds VALUES (?,?,?,?,?)',(f'REF{i+1:05d}',oid,total//2 if state=='partial_refund' else total,'pending' if state=='refund_pending' else 'succeeded','合成售后情景'))
                if state in ('delayed','exception'):
                    db.execute('INSERT INTO tickets VALUES (?,?,?,?)',(f'TKT{i+1:05d}',oid,'shipping','open'))
            counts={name:db.execute(f'SELECT count(*) FROM {name}').fetchone()[0] for name in ('users','products','orders','order_items','payments','shipments','refunds','tickets')}
            assert not db.execute('PRAGMA foreign_key_check').fetchall()
        for i in range(20):
            cases.append({'id':f'unknown-{i}','split':'test','query':f'未知设备Z{i}能不能连接火星网络',
                          'product_id':f'NONEXISTENT-{i}','expected_document_ids':[],'answerable':False})
        (work/'retrieval-cases.jsonl').write_text('\n'.join(json.dumps(case,ensure_ascii=False) for case in cases)+'\n')
        manifest={'settings':settings,'counts':counts,'knowledge':kb.status(),'retrieval_cases':len(cases),
                  'origin':'Synthetic operational data; public PQA knowledge only when public_sha256 is set',
                  'sha256':{name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ('commerce.sqlite3','knowledge.sqlite3','retrieval-cases.jsonl')}}
        (work/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        work.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(work)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=get_settings().data_dir/'portfolio')
    parser.add_argument('--products',type=int,default=320)
    parser.add_argument('--users',type=int,default=1000)
    parser.add_argument('--orders',type=int,default=10000)
    parser.add_argument('--public',type=Path)
    args=parser.parse_args()
    if not (8<=args.products<=10000 and 1<=args.users<=100000 and 10<=args.orders<=500000):parser.error('数据规模超出范围')
    print(json.dumps(build(args.output,args.products,args.users,args.orders,public=args.public),ensure_ascii=False,indent=2))

from fastapi import APIRouter

from app.commerce.store import get_commerce
from app.core.config import get_settings
from app.knowledge.store import get_knowledge

router = APIRouter()


@router.get('/identity')
def identity():
    user_id = get_settings().demo_user_id
    with get_commerce().connection() as db:
        user = db.execute('SELECT name FROM users WHERE user_id=?', (user_id,)).fetchone()
    if user is None:
        from fastapi import HTTPException
        raise HTTPException(503, '配置的演示用户不存在，请检查 DEMO_USER_ID。')
    return {'user_id': user_id, 'name': user['name'], 'fixed': True}


@router.get('/commerce/overview')
def overview():
    return {'mode': 'synthetic_demo', 'user_id': get_settings().demo_user_id,
            'counts': get_commerce().counts(),
            'products': get_commerce().products(limit=8), 'orders': get_commerce().orders(),
            'knowledge': get_knowledge().status()}

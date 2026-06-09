from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import get_settings

_settings = get_settings()


def _get_user_id_or_ip(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    return str(user_id) if user_id else get_remote_address(request)


limiter = Limiter(
    key_func=_get_user_id_or_ip,
    storage_uri=_settings.redis_url if _settings.redis_url else "memory://",
)

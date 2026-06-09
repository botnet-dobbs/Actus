from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_user_id_or_ip(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    return str(user_id) if user_id else get_remote_address(request)


limiter = Limiter(key_func=_get_user_id_or_ip)

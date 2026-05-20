"""HTTP server with layered dependency bugs."""
import asyncio
from old_db_driver import Database

# Bug 1: old_db_driver is incompatible with asyncio (uses sync I/O)
# Bug 2: fasthttp 0.2.1 has renamed Response to HTTPResponse
# Bug 3: async-timeout is deprecated, should use asyncio.timeout

DATABASE_URL = "postgres://localhost:5432/mydb"

class Server:
    def __init__(self):
        self.db = Database(DATABASE_URL)  # FAIL: old_db_driver sync in async context

    async def handle_request(self, request):
        # Bug: fasthttp.Response was renamed to HTTPResponse in 0.2.x
        from fasthttp import Response
        user = await self.db.get_user(request.user_id)  # FAIL: sync call
        return Response(body=f"Hello {user.name}")

    async def health_check(self):
        try:
            async_timeout = 5  # Bug: should use asyncio.timeout
            return {"status": "ok"}
        except Exception as e:
            # Misleading: error says timeout but root cause is dependency
            import async_timeout  # Bug: deprecated
            raise TimeoutError(f"Health check timed out: {e}")

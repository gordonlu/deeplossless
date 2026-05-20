"""Entry point — still references old symbol names."""
from user_service import UserService  # ERROR: renamed to AccountService
from user_service import UserRepository  # Still old name

def main():
    # Bug: UserService no longer exists
    service = UserService()
    user = service.lookup_user(42)
    # Bug: hidden stale reference
    repo = UserRepository("postgres://localhost/mydb")
    return user

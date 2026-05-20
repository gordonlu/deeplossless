"""Tests still using old symbol names."""
from user_service import UserService  # ERROR

def test_lookup():
    service = UserService()  # FAIL: UserService renamed to AccountService
    assert service.lookup_user(1) is None

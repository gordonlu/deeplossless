"""Account management — some symbols renamed, some not."""
class AccountService:  # was UserService, renamed
    def __init__(self):
        self.users = {}

    def lookup_user(self, user_id):
        return self.users.get(user_id)

class UserRepository:  # NOT yet renamed — should be AccountRepository
    def __init__(self, db_url):
        self.db_url = db_url

    def find_by_id(self, user_id):
        pass  # Simulated DB call

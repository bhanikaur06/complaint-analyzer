"""
Accounts and sessions for the web app. Standard library only.

What this protects, and how:

  Passwords     never stored. Only a salted scrypt hash is kept, in a format that
                records its own parameters so they can be strengthened later.
  Sessions      a random 256-bit token goes to the browser; only its SHA-256 hash is
                stored, so a leaked database cannot be replayed as logins.
  Guessing      failed logins are counted per account and per address, and a locked
                account is refused before any hashing happens.
  Enumeration   a wrong email and a wrong password give the same answer, and an
                unknown email still pays the cost of one hash, so response time does
                not reveal which accounts exist.
  Injection     every query is parameterised; user text is never joined into SQL.
  Memory        scrypt is memory-hungry by design, so at most two hashes run at once.
                Without that, a burst of logins could exhaust a small free-tier host.

What it does NOT do: send email, so there is no verification and no password reset.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

SESSION_SECONDS = 24 * 3600
MAX_ACCOUNTS = 1000

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
SCRYPT_MAXMEM = 64 * 1024 * 1024          # N=2**15, r=8 needs about 32 MiB
PBKDF2_ITERATIONS = 600_000               # fallback if scrypt is not compiled in

# At most two password hashes at a time (see the module note on memory).
_HASH_SLOTS = threading.BoundedSemaphore(2)

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,190}\.[A-Za-z]{2,24}$")
MIN_PASSWORD, MAX_PASSWORD = 8, 128
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "qwertyuiop", "iloveyou", "11111111", "00000000", "abc12345",
    "admin123", "welcome1", "welcome123", "letmein1", "letmein123", "passw0rd",
    "p@ssw0rd", "monkey123", "dragon123", "football1", "baseball1", "sunshine1",
    "princess1", "123123123", "1q2w3e4r", "q1w2e3r4", "changeme", "test1234",
    "india123", "india@123", "admin@123", "welcome@123",
}


# ---------------------------------------------------------------------------
# Where the database lives
# ---------------------------------------------------------------------------
def pick_data_dir():
    """First writable directory out of DATA_DIR, the user's home, then the temp dir.

    Deliberately outside the project folder. Uploading the project to GitHub by
    drag-and-drop ignores .gitignore, so a database inside it could end up public.
    """
    candidates = [
        os.environ.get("DATA_DIR"),
        str(Path.home() / ".complaint-analyzer"),
        os.path.join(tempfile.gettempdir(), "complaint-analyzer"),
    ]
    for c in candidates:
        if not c:
            continue
        try:
            p = Path(c)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write-test"
            probe.write_text("ok")
            probe.unlink()
            return p
        except OSError:
            continue
    raise RuntimeError("No writable folder found for the accounts database.")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def hash_password(password):
    salt = secrets.token_bytes(16)
    pw = password.encode("utf-8")
    with _HASH_SLOTS:
        if hasattr(hashlib, "scrypt"):
            dk = hashlib.scrypt(pw, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                                maxmem=SCRYPT_MAXMEM, dklen=32)
            return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${dk.hex()}"
        dk = hashlib.pbkdf2_hmac("sha256", pw, salt, PBKDF2_ITERATIONS, 32)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password, stored):
    """Constant-time check against a stored hash. Any malformed record is a failure."""
    try:
        pw = password.encode("utf-8")
        kind, *rest = stored.split("$")
        with _HASH_SLOTS:
            if kind == "scrypt":
                n, r, p, salt, want = int(rest[0]), int(rest[1]), int(rest[2]), rest[3], rest[4]
                got = hashlib.scrypt(pw, salt=bytes.fromhex(salt), n=n, r=r, p=p,
                                     maxmem=SCRYPT_MAXMEM, dklen=32).hex()
            elif kind == "pbkdf2":
                it, salt, want = int(rest[0]), rest[1], rest[2]
                got = hashlib.pbkdf2_hmac("sha256", pw, bytes.fromhex(salt), it, 32).hex()
            else:
                return False
        return hmac.compare_digest(got, want)
    except (ValueError, IndexError, AttributeError, MemoryError):
        return False


# A real hash of a throwaway password, used so that an unknown email costs the same
# time as a known one.
_DUMMY_HASH = None


def _dummy_hash():
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(12))
    return _DUMMY_HASH


# ---------------------------------------------------------------------------
# Input rules
# ---------------------------------------------------------------------------
def clean_email(raw):
    return str(raw or "").strip().lower()


def check_email(email):
    if len(email) > 254 or not EMAIL_RE.match(email):
        return "Enter a valid email address."
    return None


def check_password(password, email=""):
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    if len(password) > MAX_PASSWORD:
        return f"Use at most {MAX_PASSWORD} characters."
    low = password.lower()
    if low in COMMON_PASSWORDS or len(set(low)) < 3:
        return "That password is too common. Pick something harder to guess."
    local = email.split("@")[0]
    if email and (low == email or (len(local) >= 4 and low == local)):
        return "The password should not be the same as your email."
    return None


# ---------------------------------------------------------------------------
# Failure counting
# ---------------------------------------------------------------------------
class Throttle:
    """Sliding-window counter held in memory. Resets when the server restarts."""

    def __init__(self, limit, window_seconds):
        self.limit, self.window = limit, window_seconds
        self._hits, self._lock = {}, threading.Lock()

    def _recent(self, key, now):
        recent = [t for t in self._hits.get(key, []) if now - t < self.window]
        if recent:
            self._hits[key] = recent
        else:
            self._hits.pop(key, None)
        return recent

    def wait_seconds(self, key):
        """0 if allowed, otherwise how many seconds until the oldest hit expires."""
        now = time.time()
        with self._lock:
            recent = self._recent(key, now)
            if len(recent) >= self.limit:
                return int(self.window - (now - recent[0])) + 1
        return 0

    def add(self, key):
        now = time.time()
        with self._lock:
            recent = self._recent(key, now)
            recent.append(now)
            self._hits[key] = recent
            if len(self._hits) > 20000:     # memory guard against key flooding
                self._hits.clear()

    def clear(self, key):
        with self._lock:
            self._hits.pop(key, None)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class AuthStore:
    def __init__(self, path):
        self.path = str(path)
        with closing(self._db()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS users(
                uid TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                pw TEXT NOT NULL, created INTEGER NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions(
                token_hash TEXT PRIMARY KEY, uid TEXT NOT NULL, expires INTEGER NOT NULL)""")
        try:
            os.chmod(self.path, 0o600)       # owner only, where the OS supports it
        except OSError:
            pass
        self.login_by_email = Throttle(limit=5, window_seconds=600)
        self.login_by_addr = Throttle(limit=40, window_seconds=600)
        self.signup_by_addr = Throttle(limit=20, window_seconds=3600)

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    # ---- accounts ----
    def count_users(self):
        with closing(self._db()) as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def _get(self, email):
        with closing(self._db()) as db:
            return db.execute("SELECT uid, email, pw FROM users WHERE email = ?",
                              (email,)).fetchone()

    def create_user(self, email, password):
        """Return (uid, None) on success or (None, message) on failure."""
        err = check_email(email) or check_password(password, email)
        if err:
            return None, err
        if self.count_users() >= MAX_ACCOUNTS:
            return None, "This server has reached its account limit."
        uid = secrets.token_hex(8)        # random, so a recreated account never reuses an id
        pw = hash_password(password)
        try:
            with closing(self._db()) as db, db:
                db.execute("INSERT INTO users(uid, email, pw, created) VALUES(?,?,?,?)",
                           (uid, email, pw, int(time.time())))
        except sqlite3.IntegrityError:
            return None, "exists"
        return uid, None

    def ensure_user(self, email, password):
        """Create the account, or reset its password if the stored one differs.
        Used for the demo account defined by environment variables."""
        row = self._get(email)
        if row is None:
            return self.create_user(email, password)
        if not verify_password(password, row[2]):
            with closing(self._db()) as db, db:
                db.execute("UPDATE users SET pw = ? WHERE email = ?",
                           (hash_password(password), email))
                db.execute("DELETE FROM sessions WHERE uid = ?", (row[0],))
        return row[0], None

    # ---- sign in ----
    def login(self, email, password, addr):
        """Return a dict: {ok, token|error, status, retry}."""
        wait = max(self.login_by_email.wait_seconds(email), self.login_by_addr.wait_seconds(addr))
        if wait:
            return {"ok": False, "status": 429, "retry": wait,
                    "error": f"Too many failed attempts. Try again in {max(1, wait // 60)} minute(s)."}
        row = self._get(email) if check_email(email) is None and isinstance(password, str) \
            and len(password) <= MAX_PASSWORD else None
        # Always hash, even for an unknown email, so timing does not reveal accounts.
        good = verify_password(password if isinstance(password, str) else "",
                               row[2] if row else _dummy_hash())
        if not (row and good):
            self.login_by_email.add(email)
            self.login_by_addr.add(addr)
            return {"ok": False, "status": 401, "error": "Incorrect email or password."}
        self.login_by_email.clear(email)
        return {"ok": True, "token": self.new_session(row[0])}

    # ---- sessions ----
    @staticmethod
    def _digest(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def new_session(self, uid):
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            db.execute("INSERT INTO sessions(token_hash, uid, expires) VALUES(?,?,?)",
                       (self._digest(token), uid, now + SESSION_SECONDS))
        return token

    def user_for(self, token):
        """The signed-in user for a cookie value, or None."""
        if not token or len(token) > 100 or not re.fullmatch(r"[A-Za-z0-9_\-]+", token):
            return None
        with closing(self._db()) as db:
            row = db.execute(
                "SELECT u.uid, u.email, s.expires FROM sessions s JOIN users u ON u.uid = s.uid "
                "WHERE s.token_hash = ?", (self._digest(token),)).fetchone()
        if not row or row[2] < time.time():
            return None
        return {"uid": row[0], "email": row[1]}

    def end_session(self, token):
        if token:
            with closing(self._db()) as db, db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (self._digest(token),))

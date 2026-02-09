"""Flask web interface for the Task Manager agent — with user authentication."""

from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import logging
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

from agent import (
    DATABASE_PATH, TOOLS, SYSTEM_PROMPT, MODEL,
    ConfigError, DatabaseError, APIError, InputError,
    init_db, _require_api_key, MAX_INPUT_LENGTH,
)

log = logging.getLogger("task-agent.web")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())

# ---------------------------------------------------------------------------
# Flask-Login setup
# ---------------------------------------------------------------------------

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = None  # suppress default flash message


class User(UserMixin):
    """Lightweight user model backed by SQLite."""

    def __init__(self, id, username, email, password_hash, created_at=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at

    @staticmethod
    def get_by_id(user_id):
        try:
            conn = get_db()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            conn.close()
            if row:
                return User(**dict(row))
        except Exception:
            pass
        return None

    @staticmethod
    def get_by_username(username):
        try:
            conn = get_db()
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()
            if row:
                return User(**dict(row))
        except Exception:
            pass
        return None

    @staticmethod
    def get_by_email(email):
        try:
            conn = get_db()
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            conn.close()
            if row:
                return User(**dict(row))
        except Exception:
            pass
        return None

    @staticmethod
    def create(username, email, password):
        conn = get_db()
        try:
            pw_hash = generate_password_hash(password)
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, pw_hash),
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return User(user_id, username, email, pw_hash)
        except sqlite3.IntegrityError:
            conn.close()
            return None
        except sqlite3.Error as e:
            conn.close()
            raise DatabaseError(f"Failed to create user: {e}")


@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(int(user_id))


# ---------------------------------------------------------------------------
# Redirect unauthenticated API requests to JSON (not HTML redirect)
# ---------------------------------------------------------------------------

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required.", "error_type": "auth_error"}), 401
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db():
    """Get a database connection. Raises DatabaseError on failure."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except sqlite3.Error as e:
        log.error(f"Failed to connect to database: {e}")
        raise DatabaseError(f"Cannot connect to the database: {e}")


def execute_tool(conn, tool_name, tool_input, user_id=None):
    """Execute a tool handler — imported logic from agent.py."""
    from agent import TOOL_HANDLERS
    handler = TOOL_HANDLERS.get(tool_name)
    if handler:
        return handler(conn, tool_input, user_id=user_id)
    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Command processing
# ---------------------------------------------------------------------------


def process_command(user_message, user_id=None):
    """Send user message to Claude and execute tool calls. Returns assistant text.

    Raises:
        ConfigError  – API key missing or invalid.
        DatabaseError – DB connection or query failed.
        APIError     – Anthropic API call failed.
        InputError   – User input failed validation.
    """
    import anthropic as _anthropic
    _require_api_key()
    from agent import client

    conn = get_db()
    try:
        # Validate input (reuse agent's validator)
        from agent import validate_user_input
        user_message = validate_user_input(user_message)

        messages = [{"role": "user", "content": user_message}]
        rounds = 0
        max_rounds = 10

        while True:
            rounds += 1
            if rounds > max_rounds:
                raise APIError(
                    "The assistant got stuck in a loop. Please try rephrasing your request."
                )

            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except _anthropic.AuthenticationError:
                raise ConfigError(
                    "Invalid API key. Please check your ANTHROPIC_API_KEY."
                )
            except _anthropic.RateLimitError:
                raise APIError(
                    "Rate limit exceeded. Please wait a moment and try again."
                )
            except _anthropic.APIConnectionError:
                raise APIError(
                    "Cannot reach the AI service. Please check your internet connection."
                )
            except _anthropic.APIStatusError as e:
                log.error(f"API status error: {e.status_code}")
                raise APIError(
                    f"The AI service returned an error (HTTP {e.status_code}). Try again later."
                )
            except Exception as e:
                log.error(f"Unexpected API error: {e}")
                raise APIError(
                    "An unexpected error occurred while contacting the AI. Please try again."
                )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(conn, block.name, block.input, user_id=user_id)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "I'm not sure how to help with that."
    finally:
        conn.close()


def get_all_tasks(user_id=None):
    """Fetch all tasks for the given user. Raises DatabaseError on failure."""
    conn = get_db()
    try:
        if user_id is not None:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY id", (user_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        tasks = [dict(row) for row in rows]
        today = datetime.now().strftime("%Y-%m-%d")
        for task in tasks:
            task["overdue"] = (
                bool(task.get("due_date"))
                and task["due_date"] < today
                and task.get("status") != "completed"
            )
        return tasks
    except sqlite3.Error as e:
        log.error(f"Database error fetching tasks: {e}")
        raise DatabaseError(f"Failed to load tasks: {e}")
    finally:
        conn.close()


def get_categories(user_id=None):
    """Fetch all distinct categories for the given user. Raises DatabaseError on failure."""
    conn = get_db()
    try:
        if user_id is not None:
            rows = conn.execute(
                "SELECT DISTINCT category FROM tasks "
                "WHERE user_id = ? AND category IS NOT NULL AND category != '' "
                "ORDER BY category",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT category FROM tasks WHERE category IS NOT NULL AND category != '' ORDER BY category"
            ).fetchall()
        return [row["category"] for row in rows]
    except sqlite3.Error as e:
        log.error(f"Database error fetching categories: {e}")
        raise DatabaseError(f"Failed to load categories: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

import re

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_signup(username, email, password):
    """Validate signup fields. Returns (ok, error_message)."""
    if not username or not _USERNAME_RE.match(username):
        return False, "Username must be 3\u201330 characters (letters, numbers, underscores)."
    if not email or not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    if len(password) > 128:
        return False, "Password must be under 128 characters."
    return True, ""


# ---------------------------------------------------------------------------
# JSON error response builder
# ---------------------------------------------------------------------------

# Maps exception types to (HTTP status, error_type label)
_ERROR_MAP = {
    InputError:    (400, "input_error"),
    ConfigError:   (503, "config_error"),
    DatabaseError: (503, "database_error"),
    APIError:      (502, "api_error"),
}


def _error_response(exc):
    """Turn a known exception into a JSON error response."""
    status, error_type = _ERROR_MAP.get(type(exc), (500, "server_error"))
    return jsonify({
        "error": str(exc),
        "error_type": error_type,
    }), status


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip().lower()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"error": "Please enter both username and password."}), 400

        user = User.get_by_username(username)
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid username or password."}), 401

        login_user(user, remember=True)
        return jsonify({"ok": True, "redirect": url_for("index")})

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip().lower()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        ok, err = validate_signup(username, email, password)
        if not ok:
            return jsonify({"error": err}), 400

        # Check uniqueness
        if User.get_by_username(username):
            return jsonify({"error": "Username is already taken."}), 409
        if User.get_by_email(email):
            return jsonify({"error": "Email is already registered."}), 409

        try:
            user = User.create(username, email, password)
        except DatabaseError as e:
            return jsonify({"error": str(e)}), 500

        if not user:
            return jsonify({"error": "Could not create account. Please try again."}), 500

        login_user(user, remember=True)
        return jsonify({"ok": True, "redirect": url_for("index")})

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Main routes
# ---------------------------------------------------------------------------


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user)


@app.route("/api/health")
def api_health():
    """Health-check endpoint — tests DB connectivity and API key presence."""
    checks = {"database": "ok", "api_key": "ok"}
    status = 200

    # DB check
    try:
        conn = get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
    except Exception as e:
        checks["database"] = str(e)
        status = 503

    # API key check
    try:
        _require_api_key()
    except ConfigError as e:
        checks["api_key"] = str(e)
        status = 503

    return jsonify({"status": "healthy" if status == 200 else "unhealthy", "checks": checks}), status


@app.route("/api/tasks")
@login_required
def api_tasks():
    """Return all tasks for the current user as JSON, plus metadata."""
    try:
        return jsonify({
            "tasks": get_all_tasks(user_id=current_user.id),
            "categories": get_categories(user_id=current_user.id),
        })
    except DatabaseError as e:
        return _error_response(e)
    except Exception as e:
        log.error(f"Unexpected error in /api/tasks: {e}")
        return jsonify({"error": "Failed to load tasks. Please try again.", "error_type": "server_error"}), 500


@app.route("/api/command", methods=["POST"])
@login_required
def api_command():
    """Process a natural language command and return the response."""

    # --- Parse JSON body ---
    try:
        data = request.get_json(silent=True)
    except Exception:
        data = None

    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid request. Please send a JSON body with a 'message' field.",
            "error_type": "input_error",
        }), 400

    user_message = data.get("message", "")
    if isinstance(user_message, str):
        user_message = user_message.strip()

    # --- Validate input ---
    if not user_message:
        return jsonify({
            "error": "Please enter a command. Try something like 'add buy groceries' or 'show all tasks'.",
            "error_type": "input_error",
        }), 400

    if len(user_message) > MAX_INPUT_LENGTH:
        return jsonify({
            "error": f"Command is too long ({len(user_message)} characters). Keep it under {MAX_INPUT_LENGTH}.",
            "error_type": "input_error",
        }), 400

    # --- Process (scoped to current user) ---
    try:
        reply = process_command(user_message, user_id=current_user.id)
    except (InputError, ConfigError, DatabaseError, APIError) as e:
        return _error_response(e)
    except Exception as e:
        log.error(f"Unexpected error in /api/command: {e}")
        return jsonify({
            "error": "Something went wrong. Please try again.",
            "error_type": "server_error",
        }), 500

    # --- Build successful response (tasks may still fail) ---
    try:
        all_tasks = get_all_tasks(user_id=current_user.id)
        categories = get_categories(user_id=current_user.id)
    except DatabaseError:
        # Command succeeded but we can't refresh the task list
        all_tasks = []
        categories = []

    return jsonify({
        "reply": reply,
        "tasks": all_tasks,
        "categories": categories,
    })


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found", "error_type": "not_found"}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Method not allowed", "error_type": "method_error"}), 405


@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal server error. Please try again.", "error_type": "server_error"}), 500


# ---------------------------------------------------------------------------
# Startup — init DB when module loads (works for both gunicorn and __main__)
# ---------------------------------------------------------------------------

try:
    init_db()
except DatabaseError as e:
    log.error(f"Database init failed: {e}")
    # Don't crash — health endpoint will report the issue


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)

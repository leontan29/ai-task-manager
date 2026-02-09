from dotenv import load_dotenv
load_dotenv()
import os
import sys
import sqlite3
import logging
from datetime import datetime

import anthropic

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("task-agent")

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when the app is misconfigured (missing API key, etc.)."""


class DatabaseError(Exception):
    """Raised when a database operation fails."""


class APIError(Exception):
    """Raised when the Anthropic API call fails."""


class InputError(Exception):
    """Raised when user input fails validation."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 1000          # characters
MAX_TOOL_ROUNDS = 10             # safety cap on tool-use loops
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_STATUSES = {"pending", "in_progress", "completed"}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.db")
MODEL = "claude-sonnet-4-5-20250929"

# Deferred API key check — don't crash at import time so Flask can still
# serve a helpful error page.  CLI mode calls _require_api_key() in main().
_api_key = os.environ.get("ANTHROPIC_API_KEY")
client = None

def _require_api_key():
    """Initialise the Anthropic client, raising ConfigError if the key is absent."""
    global client, _api_key
    if client is not None:
        return
    _api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not _api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. "
            "Create a .env file with your key or run: export ANTHROPIC_API_KEY='your-key-here'"
        )
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        raise ConfigError(f"Failed to initialise Anthropic client: {e}")

# Try to init eagerly (best-effort) so tools/prompt stay importable
try:
    _require_api_key()
except ConfigError:
    log.warning("ANTHROPIC_API_KEY not set — API calls will fail until it is configured.")

SYSTEM_PROMPT = """You are a helpful task manager assistant. The user will give you natural language \
commands to manage their to-do list. Use the provided tools to add, list, update, complete, or \
delete tasks.

When listing tasks, format the results in a clear, readable way.
When the user's intent is unclear, ask for clarification rather than guessing.
Always confirm actions you take (e.g., "I've added the task..." or "Here are your tasks...").

IMPORTANT — Due dates: The user may specify due dates in natural language such as "tomorrow", \
"next Friday", "in 3 days", "end of week", etc. You MUST convert these to YYYY-MM-DD format \
before passing them to the tools. Today's date is {today}.

IMPORTANT — Categories: The user may assign a category/tag to tasks using phrases like \
"category shopping", "under work", "tag personal", "in the errands category", etc. \
Pass the category as a short lowercase label (e.g. "shopping", "work", "personal", "health"). \
If the user doesn't specify a category, omit it — do NOT default to one.

When listing tasks, you can filter by category using the category parameter. \
You can also sort results by due_date to show the most urgent tasks first.""".format(
    today=datetime.now().strftime("%Y-%m-%d")
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def init_db():
    """Create the tasks table if it doesn't exist and return the connection.

    Raises DatabaseError on failure.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        raise DatabaseError(f"Cannot open database at {DATABASE_PATH}: {e}")

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                due_date TEXT DEFAULT NULL,
                category TEXT DEFAULT NULL
            )
            """
        )
        conn.commit()
    except sqlite3.Error as e:
        conn.close()
        raise DatabaseError(f"Failed to create tasks table: {e}")

    # Migrate: add category column if it doesn't exist (for existing databases)
    try:
        conn.execute("SELECT category FROM tasks LIMIT 1")
    except sqlite3.OperationalError:
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN category TEXT DEFAULT NULL")
            conn.commit()
        except sqlite3.Error as e:
            conn.close()
            raise DatabaseError(f"Failed to migrate database (add category column): {e}")

    return conn


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def validate_user_input(text):
    """Validate raw user input. Raises InputError if invalid."""
    if not text or not text.strip():
        raise InputError("Please enter a command. Try something like 'add buy groceries' or 'show all tasks'.")
    text = text.strip()
    if len(text) > MAX_INPUT_LENGTH:
        raise InputError(
            f"Command is too long ({len(text)} characters). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )
    return text


def validate_priority(priority):
    """Return a valid priority or raise InputError."""
    if priority and priority not in VALID_PRIORITIES:
        raise InputError(
            f"Invalid priority '{priority}'. Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
        )
    return priority


def validate_status(status):
    """Return a valid status or raise InputError."""
    if status and status not in VALID_STATUSES:
        raise InputError(
            f"Invalid status '{status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )
    return status


def validate_due_date(due_date):
    """Validate YYYY-MM-DD format. Raises InputError if malformed."""
    if not due_date:
        return due_date
    try:
        datetime.strptime(due_date, "%Y-%m-%d")
    except ValueError:
        raise InputError(
            f"Invalid due date '{due_date}'. Expected format: YYYY-MM-DD (e.g. 2026-03-15)"
        )
    return due_date


def validate_category(category):
    """Validate category label. Raises InputError if too long."""
    if not category:
        return category
    if len(category) > 50:
        raise InputError("Category name is too long. Please keep it under 50 characters.")
    return category.strip().lower()


# ---------------------------------------------------------------------------
# Tool definitions (sent to the Anthropic API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "add_task",
        "description": (
            "Add a new task to the task list. Use this when the user wants to create, "
            "add, or remember a new task or to-do item."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title of the task",
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer description",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Priority level (defaults to 'medium')",
                },
                "due_date": {
                    "type": "string",
                    "description": (
                        "Due date in YYYY-MM-DD format. The assistant must convert "
                        "natural language dates (e.g. 'tomorrow', 'next Friday') to "
                        "this format before calling this tool."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category or tag for the task, as a short lowercase "
                        "label (e.g. 'shopping', 'work', 'personal', 'health'). "
                        "Omit if the user doesn't specify one."
                    ),
                },
            },
            "required": ["title"],
        },
    },
    {
        "name": "list_tasks",
        "description": (
            "List tasks from the task list. Supports optional filtering by status, "
            "priority, and/or category. Supports optional sorting by due_date or priority."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                    "description": "Filter by status. Omit to show all.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Filter by priority. Omit to show all.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Filter by category (e.g. 'shopping', 'work'). Omit to show all."
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["due_date", "priority", "created_at"],
                    "description": (
                        "Sort results by this field. Defaults to 'id' if omitted. "
                        "'due_date' puts tasks with nearest due dates first (nulls last). "
                        "'priority' orders urgent > high > medium > low."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_task",
        "description": (
            "Update fields of an existing task. The user refers to the task by its "
            "numeric ID. Only include fields that are being changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The numeric ID of the task to update",
                },
                "title": {"type": "string", "description": "New title"},
                "description": {"type": "string", "description": "New description"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "New priority level",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                    "description": "New status",
                },
                "due_date": {
                    "type": "string",
                    "description": "New due date in YYYY-MM-DD format",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "New category label, or empty string to remove the category"
                    ),
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": (
            "Mark a task as completed. The user refers to the task by its numeric ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The numeric ID of the task to complete",
                }
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "delete_task",
        "description": (
            "Permanently delete a task. The user refers to the task by its numeric ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "The numeric ID of the task to delete",
                }
            },
            "required": ["task_id"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------


def handle_add_task(conn, data):
    try:
        title = data.get("title", "").strip()
        if not title:
            return "Error: task title cannot be empty."
        if len(title) > 200:
            return "Error: task title is too long (max 200 characters)."

        description = data.get("description", "")
        priority = data.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            return f"Error: invalid priority '{priority}'. Use: {', '.join(sorted(VALID_PRIORITIES))}"

        due_date = data.get("due_date")
        if due_date:
            try:
                datetime.strptime(due_date, "%Y-%m-%d")
            except ValueError:
                return f"Error: invalid due date format '{due_date}'. Use YYYY-MM-DD."

        category = data.get("category")
        if category and len(category) > 50:
            return "Error: category name is too long (max 50 characters)."

        cursor = conn.execute(
            "INSERT INTO tasks (title, description, priority, due_date, category) VALUES (?, ?, ?, ?, ?)",
            (title, description, priority, due_date, category),
        )
        conn.commit()
        task_id = cursor.lastrowid
        result = f"Task added (ID {task_id}): '{title}' | priority: {priority}"
        if due_date:
            result += f" | due: {due_date}"
        if category:
            result += f" | category: {category}"
        return result
    except sqlite3.Error as e:
        log.error(f"Database error in add_task: {e}")
        return f"Database error while adding task: {e}"
    except Exception as e:
        log.error(f"Unexpected error in add_task: {e}")
        return f"Error adding task: {e}"


def handle_list_tasks(conn, data):
    try:
        query = "SELECT * FROM tasks"
        conditions = []
        params = []

        if "status" in data:
            conditions.append("status = ?")
            params.append(data["status"])
        if "priority" in data:
            conditions.append("priority = ?")
            params.append(data["priority"])
        if "category" in data:
            conditions.append("LOWER(category) = LOWER(?)")
            params.append(data["category"])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Sorting
        sort_by = data.get("sort_by")
        if sort_by == "due_date":
            query += " ORDER BY (due_date IS NULL), due_date ASC, id ASC"
        elif sort_by == "priority":
            query += (
                " ORDER BY CASE priority "
                "WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 "
                "WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END, id ASC"
            )
        elif sort_by == "created_at":
            query += " ORDER BY created_at DESC, id DESC"
        else:
            query += " ORDER BY id"

        rows = conn.execute(query, params).fetchall()

        if not rows:
            return "No tasks found."

        lines = []
        for row in rows:
            line = (
                f"  [{row['id']}] {row['title']} "
                f"| priority: {row['priority']} | status: {row['status']}"
            )
            if row["due_date"]:
                line += f" | due: {row['due_date']}"
            if row["category"]:
                line += f" | category: {row['category']}"
            lines.append(line)

        return f"Found {len(rows)} task(s):\n" + "\n".join(lines)
    except sqlite3.Error as e:
        log.error(f"Database error in list_tasks: {e}")
        return f"Database error while listing tasks: {e}"
    except Exception as e:
        log.error(f"Unexpected error in list_tasks: {e}")
        return f"Error listing tasks: {e}"


def handle_update_task(conn, data):
    try:
        task_id = data.get("task_id")
        if task_id is None:
            return "Error: task_id is required."

        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return f"No task found with ID {task_id}."

        updatable = ["title", "description", "priority", "status", "due_date", "category"]
        fields = []
        values = []
        for field in updatable:
            if field in data:
                val = data[field]
                # Validate fields
                if field == "title" and (not val or not val.strip()):
                    return "Error: title cannot be empty."
                if field == "priority" and val not in VALID_PRIORITIES:
                    return f"Error: invalid priority '{val}'."
                if field == "status" and val not in VALID_STATUSES:
                    return f"Error: invalid status '{val}'."
                if field == "due_date" and val:
                    try:
                        datetime.strptime(val, "%Y-%m-%d")
                    except ValueError:
                        return f"Error: invalid due date format '{val}'. Use YYYY-MM-DD."
                if field == "category" and val == "":
                    val = None
                fields.append(f"{field} = ?")
                values.append(val)

        if not fields:
            return "No fields to update."

        values.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()
        return f"Task {task_id} updated successfully."
    except sqlite3.Error as e:
        log.error(f"Database error in update_task: {e}")
        return f"Database error while updating task: {e}"
    except Exception as e:
        log.error(f"Unexpected error in update_task: {e}")
        return f"Error updating task: {e}"


def handle_complete_task(conn, data):
    try:
        task_id = data.get("task_id")
        if task_id is None:
            return "Error: task_id is required."

        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return f"No task found with ID {task_id}."

        if row["status"] == "completed":
            return f"Task {task_id} is already completed."

        conn.execute(
            "UPDATE tasks SET status = 'completed' WHERE id = ?", (task_id,)
        )
        conn.commit()
        return f"Task {task_id} marked as completed: '{row['title']}'"
    except sqlite3.Error as e:
        log.error(f"Database error in complete_task: {e}")
        return f"Database error while completing task: {e}"
    except Exception as e:
        log.error(f"Unexpected error in complete_task: {e}")
        return f"Error completing task: {e}"


def handle_delete_task(conn, data):
    try:
        task_id = data.get("task_id")
        if task_id is None:
            return "Error: task_id is required."

        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return f"No task found with ID {task_id}."

        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        return f"Task {task_id} deleted: '{row['title']}'"
    except sqlite3.Error as e:
        log.error(f"Database error in delete_task: {e}")
        return f"Database error while deleting task: {e}"
    except Exception as e:
        log.error(f"Unexpected error in delete_task: {e}")
        return f"Error deleting task: {e}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "add_task": handle_add_task,
    "list_tasks": handle_list_tasks,
    "update_task": handle_update_task,
    "complete_task": handle_complete_task,
    "delete_task": handle_delete_task,
}


def execute_tool(conn, tool_name, tool_input):
    handler = TOOL_HANDLERS.get(tool_name)
    if handler:
        return handler(conn, tool_input)
    log.warning(f"Unknown tool requested: {tool_name}")
    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Anthropic API integration
# ---------------------------------------------------------------------------


def process_user_input(conn, user_message):
    """Send the user's message to Claude and execute any tool calls.

    Raises:
        ConfigError  – API key not set.
        APIError     – API call failed after retries.
        InputError   – User input too long / empty.
    """
    # Validate
    user_message = validate_user_input(user_message)

    # Ensure client is ready
    _require_api_key()

    messages = [{"role": "user", "content": user_message}]
    rounds = 0

    while True:
        rounds += 1
        if rounds > MAX_TOOL_ROUNDS:
            log.error(f"Tool-use loop exceeded {MAX_TOOL_ROUNDS} rounds — aborting.")
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
        except anthropic.AuthenticationError:
            raise ConfigError(
                "Invalid API key. Please check your ANTHROPIC_API_KEY and try again."
            )
        except anthropic.RateLimitError:
            raise APIError(
                "Rate limit exceeded. Please wait a moment and try again."
            )
        except anthropic.APIConnectionError:
            raise APIError(
                "Cannot reach the Anthropic API. Please check your internet connection."
            )
        except anthropic.APIStatusError as e:
            log.error(f"Anthropic API status error: {e.status_code} — {e.message}")
            raise APIError(
                f"The AI service returned an error (HTTP {e.status_code}). Please try again later."
            )
        except Exception as e:
            log.error(f"Unexpected API error: {e}")
            raise APIError(
                "An unexpected error occurred while contacting the AI service. Please try again."
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(conn, block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Extract the final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "I'm not sure how to help with that. Try something like 'add buy groceries' or 'show all tasks'."


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def main():
    print("Task Manager CLI")
    print("Type your commands in natural language. Type 'quit' or 'exit' to stop.")
    print("Examples: 'add buy groceries', 'show all tasks', 'complete task 3'")
    print("-" * 50)

    # Fail fast in CLI mode if key is missing
    try:
        _require_api_key()
    except ConfigError as e:
        print(f"\nConfiguration error: {e}")
        sys.exit(1)

    try:
        conn = init_db()
    except DatabaseError as e:
        print(f"\nDatabase error: {e}")
        sys.exit(1)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            try:
                response = process_user_input(conn, user_input)
                print(f"\nAssistant: {response}")
            except InputError as e:
                print(f"\nInput error: {e}")
            except ConfigError as e:
                print(f"\nConfiguration error: {e}")
            except APIError as e:
                print(f"\nAPI error: {e}")
            except DatabaseError as e:
                print(f"\nDatabase error: {e}")
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                print(f"\nUnexpected error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

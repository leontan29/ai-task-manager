# Task Manager

An AI-powered task manager that understands plain English — built with Claude and Flask.

<!-- TODO: Add your demo GIF here -->
<!-- ![Demo](docs/demo.gif) -->

---

## Features

- **Natural Language Commands** — Type `add buy groceries due tomorrow category shopping` and it just works
- **Smart Date Parsing** — Say "next Friday", "in 3 days", or "end of week" — Claude converts them automatically
- **Categories & Tags** — Organize tasks with labels like `shopping`, `work`, `personal` — each gets a unique color
- **Priority Levels** — `low`, `medium`, `high`, `urgent` with color-coded badges
- **Overdue Detection** — Past-due tasks are flagged in red so nothing slips through
- **Filter & Sort** — Narrow by status, category, or priority; sort by due date or importance
- **Dark UI** — Clean, modern interface that looks good on desktop and mobile
- **Mobile-Responsive** — Thumb-friendly buttons, stacked layout, and safe-area support for notched phones
- **Error Recovery** — Retry buttons, health checks, offline detection — the app never just crashes on you

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI** | [Claude claude-sonnet-4-5-20250929](https://docs.anthropic.com/en/docs/about-claude/models) via tool use |
| **Backend** | Python, Flask, SQLite |
| **Frontend** | Vanilla HTML/CSS/JS (no build step) |
| **Production** | Gunicorn, Render |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser                          │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐   │
│  │  Command   │  │  Filters   │  │  Task List   │   │
│  │  Input     │  │  & Sort    │  │  (live)      │   │
│  └─────┬─────┘  └────────────┘  └──────────────┘   │
└────────┼────────────────────────────────────────────┘
         │ POST /api/command
         ▼
┌─────────────────────────────────────────────────────┐
│                  Flask (app.py)                      │
│                                                     │
│  /              → serves index.html                 │
│  /api/tasks     → returns all tasks as JSON         │
│  /api/command   → validates input, calls agent      │
│  /api/health    → checks DB + API key               │
└─────────┬───────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│                 Agent (agent.py)                     │
│                                                     │
│  ┌──────────┐    ┌───────────────┐    ┌──────────┐  │
│  │ Validate │───▶│ Claude API    │───▶│ Execute  │  │
│  │ Input    │    │ (tool use)    │    │ Tools    │  │
│  └──────────┘    └──────┬────────┘    └────┬─────┘  │
│                         │                  │        │
│                         │  ┌───────────────┘        │
│                         ▼  ▼                        │
│                    ┌──────────┐                      │
│                    │ SQLite   │                      │
│                    │ tasks.db │                      │
│                    └──────────┘                      │
└─────────────────────────────────────────────────────┘
```

**How a command flows:**

1. You type `add buy milk due tomorrow category shopping`
2. Flask validates length and format, forwards to the agent
3. The agent sends your message to Claude with 5 tool definitions
4. Claude interprets intent, converts "tomorrow" to `2026-02-10`, and calls `add_task`
5. The tool handler inserts a row into SQLite and returns a confirmation
6. Claude formulates a friendly response: *"I've added 'buy milk' to your list..."*
7. The UI updates the task list, showing the new card with a teal `shopping` badge

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/your-username/task-agent.git
cd task-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Run

```bash
# Web interface (recommended)
python app.py
# → Open http://localhost:5000

# CLI mode (no browser needed)
python agent.py
```

---

## Usage Examples

### Adding tasks

```
add buy groceries
add finish report with high priority due Friday category work
add call dentist due tomorrow
```

### Viewing & filtering

```
show all tasks
show my shopping tasks
list urgent tasks sorted by due date
what's overdue?
```

### Updating & completing

```
mark task 3 as done
change task 2 priority to urgent
move task 5 to the work category
update task 1 title to buy organic groceries
```

### Deleting

```
delete task 4
remove task 7
```

<!-- TODO: Add screenshots here -->
<!-- ![Adding a task](docs/add-task.png) -->
<!-- ![Filtered view](docs/filtered-view.png) -->

---

## Deploying to Render

The project includes `render.yaml` for one-click deployment:

1. Push your repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**
3. Connect your repo — Render auto-detects `render.yaml`
4. Set `ANTHROPIC_API_KEY` in the environment variables panel
5. Deploy

Or deploy manually as a **Web Service**:

| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| Environment | `ANTHROPIC_API_KEY` = your key |

> **Note:** Render uses an ephemeral filesystem — the SQLite database resets on each deploy. For persistent storage, swap to PostgreSQL.

---

## Project Structure

```
task-agent/
├── agent.py            # AI agent: tools, handlers, Claude integration
├── app.py              # Flask web server and API routes
├── templates/
│   └── index.html      # Single-page UI (CSS + JS embedded)
├── requirements.txt    # Pinned Python dependencies
├── render.yaml         # Render deployment blueprint
├── Procfile            # Gunicorn start command
├── .env                # API key (not committed)
└── .gitignore
```

---

## Future Improvements

- [ ] **Persistent database** — swap SQLite for PostgreSQL on Render
- [ ] **Conversation memory** — multi-turn context so Claude remembers prior messages
- [ ] **Recurring tasks** — "remind me every Monday to review PRs"
- [ ] **Multiple lists / projects** — group tasks into boards
- [ ] **User authentication** — personal task lists with login
- [ ] **Drag-and-drop reordering** — manual task prioritization
- [ ] **Dark / light theme toggle**
- [ ] **Export** — download tasks as CSV or JSON
- [ ] **Notifications** — browser push alerts for upcoming due dates
- [ ] **Keyboard shortcuts** — power-user navigation (`/` to focus input, `j`/`k` to move)

---

## License

MIT

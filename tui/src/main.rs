use std::io::{self, Stdout};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use ratatui::Terminal;

#[derive(Parser, Debug)]
#[command(name = "hpc-assistant-tui")]
#[command(about = "Prototype Rust TUI for the hpc-assistant backend")]
struct Args {
    #[arg(long, env = "HPC_ASSISTANT_BACKEND_COMMAND", default_value = "hpc-assistant-backend")]
    backend_command: String,
    #[arg(long, env = "HPC_ASSISTANT_REPO_ROOT", default_value = ".")]
    repo_root: String,
    #[arg(long, env = "HPC_ASSISTANT_GOAL")]
    goal: Option<String>,
}

struct Panel {
    title: &'static str,
    argv: Vec<String>,
    body: String,
    status: String,
}

struct App {
    backend_command: String,
    selected: usize,
    panels: Vec<Panel>,
}

impl App {
    fn new(backend_command: String, repo_root: String, goal: Option<String>) -> Self {
        let mut assist_argv = vec![
            String::from("assist"),
            String::from("--repo"),
            repo_root,
        ];
        if let Some(goal) = goal {
            assist_argv.push(String::from("--goal"));
            assist_argv.push(goal);
        }

        Self {
            backend_command,
            selected: 0,
            panels: vec![
                Panel {
                    title: "Assist",
                    argv: assist_argv,
                    body: String::from("Press r to initialize the repo-aware assistant session."),
                    status: String::from("idle"),
                },
                Panel {
                    title: "Doctor",
                    argv: vec![String::from("doctor")],
                    body: String::from("Press r to load the doctor report."),
                    status: String::from("idle"),
                },
                Panel {
                    title: "Cluster",
                    argv: vec![String::from("discover-cluster")],
                    body: String::from("Press r to discover cluster topology."),
                    status: String::from("idle"),
                },
                Panel {
                    title: "Environment",
                    argv: vec![String::from("discover-env")],
                    body: String::from("Press r to inspect the module environment."),
                    status: String::from("idle"),
                },
            ],
        }
    }

    fn refresh_current(&mut self) {
        if let Some(panel) = self.panels.get_mut(self.selected) {
            let label = panel.argv.join(" ");
            panel.status = format!("running {label}");
            match run_backend_command(&self.backend_command, &panel.argv) {
                Ok(output) => {
                    panel.body = prettify_json(&output);
                    panel.status = format!("completed {label}");
                }
                Err(error) => {
                    panel.body = format!("backend command failed:\n\n{error:#}");
                    panel.status = format!("failed {label}");
                }
            }
        }
    }
}

fn run_backend_command(backend_command: &str, argv: &[String]) -> Result<String> {
    let output = Command::new(backend_command)
        .args(argv)
        .output()
        .with_context(|| format!("failed to start backend command `{backend_command}`"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let joined = argv.join(" ");
        anyhow::bail!(
            "`{backend_command} {joined}` exited with {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout.trim(),
            stderr.trim(),
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn prettify_json(raw: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut terminal = setup_terminal()?;
    let result = run_app(
        &mut terminal,
        App::new(args.backend_command, args.repo_root, args.goal),
    );
    restore_terminal(terminal)?;
    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: App) -> Result<()> {
    app.refresh_current();

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(2),
                    Constraint::Min(1),
                    Constraint::Length(2),
                ])
                .split(area);

            let title = Paragraph::new("HPC Assistant Prototype")
                .style(Style::default().fg(Color::Cyan))
                .block(Block::default().borders(Borders::ALL).title("Overview"));
            frame.render_widget(title, layout[0]);

            let tab_titles: Vec<Line> = app
                .panels
                .iter()
                .map(|panel| Line::from(panel.title))
                .collect();
            let tabs = Tabs::new(tab_titles)
                .select(app.selected)
                .highlight_style(Style::default().fg(Color::Yellow))
                .block(Block::default().borders(Borders::ALL).title("Views"));
            frame.render_widget(tabs, layout[1]);

            let panel = &app.panels[app.selected];
            let body = Paragraph::new(Text::from(panel.body.clone()))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("{} Output", panel.title)),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(body, layout[2]);

            let footer = Paragraph::new(Line::from(vec![
                "1/2/3/4".bold(),
                " switch view  ".into(),
                "r".bold(),
                " refresh  ".into(),
                "q".bold(),
                " quit  ".into(),
                "status: ".into(),
                panel.status.clone().yellow(),
            ]))
            .block(Block::default().borders(Borders::ALL).title("Controls"));
            frame.render_widget(footer, layout[3]);
        })?;

        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('1') => app.selected = 0,
                    KeyCode::Char('2') => app.selected = 1,
                    KeyCode::Char('3') => app.selected = 2,
                    KeyCode::Char('4') => app.selected = 3,
                    KeyCode::Char('r') => app.refresh_current(),
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

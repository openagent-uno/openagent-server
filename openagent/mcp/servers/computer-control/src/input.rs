//! Input actions (keyboard + mouse) backed by enigo.
//! enigo uses: CGEvent (macOS), XTest or Wayland portal (Linux), SendInput (Windows).

use anyhow::{Context, Result, anyhow};
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Keyboard, Mouse, Settings};

#[cfg(target_os = "macos")]
pub const MAC_ACCESSIBILITY_HINT: &str =
    "macOS Accessibility permission required. Open System Settings → Privacy & Security → Accessibility and enable 'openagent', then restart the app.";

#[cfg(target_os = "macos")]
fn is_accessibility_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}").to_lowercase();
    s.contains("accessibility") || s.contains("not trusted") || s.contains("axiserror")
}

use crate::keys;

pub struct InputController {
    enigo: Enigo,
}

impl InputController {
    pub fn new() -> Result<Self> {
        let enigo = Enigo::new(&Settings::default()).map_err(|e| {
            let err: anyhow::Error = e.into();
            #[cfg(target_os = "macos")]
            if is_accessibility_error(&err) {
                return anyhow!(MAC_ACCESSIBILITY_HINT);
            }
            err.context("enigo init failed")
        })?;
        Ok(Self { enigo })
    }

    pub fn type_text(&mut self, text: &str) -> Result<()> {
        self.enigo.text(text).context("enigo.text")?;
        Ok(())
    }

    pub fn key_chord(&mut self, spec: &str) -> Result<()> {
        let keys = keys::parse(spec).map_err(|e| anyhow!(e))?;
        if keys.is_empty() {
            return Err(anyhow!("empty key spec"));
        }
        // Press in order, release in reverse order.
        for k in &keys {
            self.enigo
                .key(*k, Direction::Press)
                .context("key press")?;
        }
        for k in keys.iter().rev() {
            self.enigo
                .key(*k, Direction::Release)
                .context("key release")?;
        }
        Ok(())
    }

    pub fn mouse_move(&mut self, x: i32, y: i32) -> Result<()> {
        self.enigo
            .move_mouse(x, y, Coordinate::Abs)
            .context("move_mouse")?;
        Ok(())
    }

    pub fn cursor_position(&self) -> Result<(i32, i32)> {
        self.enigo.location().context("enigo.location")
    }

    pub fn left_click(&mut self, at: Option<(i32, i32)>) -> Result<()> {
        if let Some((x, y)) = at {
            self.mouse_move(x, y)?;
        }
        self.enigo
            .button(Button::Left, Direction::Click)
            .context("left click")?;
        Ok(())
    }

    pub fn right_click(&mut self, at: Option<(i32, i32)>) -> Result<()> {
        if let Some((x, y)) = at {
            self.mouse_move(x, y)?;
        }
        self.enigo
            .button(Button::Right, Direction::Click)
            .context("right click")?;
        Ok(())
    }

    pub fn middle_click(&mut self, at: Option<(i32, i32)>) -> Result<()> {
        if let Some((x, y)) = at {
            self.mouse_move(x, y)?;
        }
        self.enigo
            .button(Button::Middle, Direction::Click)
            .context("middle click")?;
        Ok(())
    }

    pub fn double_click(&mut self, at: Option<(i32, i32)>) -> Result<()> {
        if let Some((x, y)) = at {
            self.mouse_move(x, y)?;
        }
        self.enigo.button(Button::Left, Direction::Click)?;
        self.enigo.button(Button::Left, Direction::Click)?;
        Ok(())
    }

    pub fn left_click_drag(&mut self, to: (i32, i32)) -> Result<()> {
        self.enigo
            .button(Button::Left, Direction::Press)
            .context("drag press")?;
        self.mouse_move(to.0, to.1)?;
        self.enigo
            .button(Button::Left, Direction::Release)
            .context("drag release")?;
        Ok(())
    }

    /// Scroll with direction ("up"|"down"|"left"|"right") and optional amount (pixels).
    /// Default amount: 300 (matches TS behavior).
    pub fn scroll(&mut self, at: (i32, i32), direction: &str, amount: Option<u32>) -> Result<()> {
        self.mouse_move(at.0, at.1)?;
        let amt = amount.unwrap_or(300) as i32;
        match direction.to_ascii_lowercase().as_str() {
            "up" => self.enigo.scroll(-amt, Axis::Vertical)?,
            "down" => self.enigo.scroll(amt, Axis::Vertical)?,
            "left" => self.enigo.scroll(-amt, Axis::Horizontal)?,
            "right" => self.enigo.scroll(amt, Axis::Horizontal)?,
            other => return Err(anyhow!("invalid scroll direction: {other}")),
        }
        Ok(())
    }
}

/// Parse the scroll `text` argument from the MCP call, e.g. "down" or "down:500".
pub fn parse_scroll_text(text: &str) -> Result<(&str, Option<u32>)> {
    let mut parts = text.splitn(2, ':');
    let dir = parts
        .next()
        .ok_or_else(|| anyhow!("scroll direction required"))?;
    if dir.is_empty() {
        return Err(anyhow!("scroll direction required"));
    }
    let amount = match parts.next() {
        None => None,
        Some(s) => {
            let n: u32 = s
                .parse()
                .map_err(|_| anyhow!("invalid scroll amount: {s}"))?;
            if n == 0 {
                return Err(anyhow!("invalid scroll amount: {s}"));
            }
            Some(n)
        }
    };
    Ok((dir, amount))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_text_parses_direction_only() {
        assert_eq!(parse_scroll_text("down").unwrap(), ("down", None));
        assert_eq!(parse_scroll_text("up").unwrap(), ("up", None));
    }

    #[test]
    fn scroll_text_parses_direction_and_amount() {
        assert_eq!(parse_scroll_text("down:500").unwrap(), ("down", Some(500)));
        assert_eq!(parse_scroll_text("left:1").unwrap(), ("left", Some(1)));
    }

    #[test]
    fn scroll_text_rejects_bad_amount() {
        assert!(parse_scroll_text("down:abc").is_err());
        assert!(parse_scroll_text("down:0").is_err());
        assert!(parse_scroll_text("down:-5").is_err());
    }

    #[test]
    fn scroll_text_rejects_empty_direction() {
        assert!(parse_scroll_text("").is_err());
        assert!(parse_scroll_text(":500").is_err());
    }
}

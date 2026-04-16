//! The single `computer` MCP tool. Dispatcher over capture + input.
//!
//! Tool surface is byte-identical to the Node implementation at
//! ../src/tools/computer.ts — action enum, parameter names, scroll syntax,
//! coordinate-scaling behavior, crosshair overlay.

use anyhow::{Context, Result, anyhow};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars,
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{capture, input, scaling};

/// All actions supported by the `computer` tool.
/// Names are snake_case to match the MCP protocol surface (= TS enum values).
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Press a key or key-combination on the keyboard.
    Key,
    /// Type a string of text on the keyboard.
    Type,
    /// Move the cursor to a specified (x, y) pixel coordinate on the screen.
    MouseMove,
    /// Click the left mouse button.
    LeftClick,
    /// Click and drag the cursor to a specified (x, y) pixel coordinate.
    LeftClickDrag,
    /// Click the right mouse button.
    RightClick,
    /// Click the middle mouse button.
    MiddleClick,
    /// Double-click the left mouse button.
    DoubleClick,
    /// Scroll the screen in a specified direction.
    Scroll,
    /// Take a screenshot of the screen.
    GetScreenshot,
    /// Get the current (x, y) pixel coordinate of the cursor on the screen.
    GetCursorPosition,
}

/// Input parameters for the `computer` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ComputerArgs {
    /// The action to perform. The available actions are:
    /// * key: Press a key or key-combination on the keyboard.
    /// * type: Type a string of text on the keyboard.
    /// * get_cursor_position: Get the current (x, y) pixel coordinate of the cursor on the screen.
    /// * mouse_move: Move the cursor to a specified (x, y) pixel coordinate on the screen.
    /// * left_click: Click the left mouse button. If coordinate is provided, moves to that position first.
    /// * left_click_drag: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
    /// * right_click: Click the right mouse button. If coordinate is provided, moves to that position first.
    /// * middle_click: Click the middle mouse button. If coordinate is provided, moves to that position first.
    /// * double_click: Double-click the left mouse button. If coordinate is provided, moves to that position first.
    /// * scroll: Scroll the screen in a specified direction. Requires coordinate (moves there first) and text parameter with direction: "up", "down", "left", or "right". Optionally append ":N" to scroll N pixels (default 300), e.g. "down:500".
    /// * get_screenshot: Take a screenshot of the screen.
    pub action: Action,
    /// `(x, y)`: The x (pixels from the left edge) and y (pixels from the top edge) coordinates in API image space; scaled to logical screen by the server.
    #[serde(default)]
    pub coordinate: Option<[i32; 2]>,
    /// Text to type or key command to execute.
    #[serde(default)]
    pub text: Option<String>,
}

// NOTE: The tool description is inlined directly in the #[tool(description = "...")] attribute
// below because the rmcp #[tool] macro only accepts string literals (not const paths).
// Do NOT edit the wording without discussion — it shapes Claude's prompting.

/// The rmcp server struct that holds the tool router and shared input controller.
///
/// The `InputController` is lazily initialized on the first tool call that needs it,
/// so that the MCP server can start and list tools even when Accessibility permission
/// has not yet been granted (the permission check only happens inside enigo::new()).
#[derive(Clone)]
pub struct ComputerControlServer {
    tool_router: ToolRouter<Self>,
    /// Lazily initialized; `None` until the first action that needs input control.
    input: Arc<Mutex<Option<input::InputController>>>,
}

#[tool_router(router = tool_router)]
impl ComputerControlServer {
    /// Create a new server instance. Does NOT initialize enigo yet.
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            input: Arc::new(Mutex::new(None)),
        }
    }

    /// The single computer tool — dispatches all 11 actions.
    #[tool(description = "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Always prefer using keyboard shortcuts rather than clicking, where possible.\n* If you see boxes with two letters in them, typing these letters will click that element. Use this instead of other shortcuts or clicking, where possible.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try taking another screenshot.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n\nUsing the crosshair:\n* Screenshots show a red crosshair at the current cursor position.\n* After clicking, check where the crosshair appears vs your target. If it missed, adjust coordinates proportionally to the distance - start with large adjustments and refine. Avoid small incremental changes when the crosshair is far from the target (distances are often further than you expect).\n* Consider display dimensions when estimating positions. E.g. if it's 90% to the bottom of the screen, the coordinates should reflect this.")]
    pub async fn computer(&self, params: Parameters<ComputerArgs>) -> CallToolResult {
        let args = params.0;
        match self.dispatch(args).await {
            Ok(result) => result,
            Err(e) => CallToolResult::error(vec![Content::text(format!("{e:#}"))]),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ComputerControlServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "openagent-computer-control",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Cross-platform mouse/keyboard/screenshot MCP for desktop GUI control.",
            )
    }
}

impl ComputerControlServer {
    async fn dispatch(&self, args: ComputerArgs) -> Result<CallToolResult> {
        // Scale coordinate from API space to logical screen space, if given.
        let logical_coord: Option<(i32, i32)> = match args.coordinate {
            None => None,
            Some([ax, ay]) => {
                let (lw, lh) = self.logical_display_size()?;
                let (lx, ly) = scaling::api_to_logical(ax, ay, lw, lh);
                if lx < 0 || lx >= lw as i32 || ly < 0 || ly >= lh as i32 {
                    return Err(anyhow!(
                        "Coordinates ({lx}, {ly}) are outside display bounds of {lw}x{lh}"
                    ));
                }
                Some((lx, ly))
            }
        };

        let mut input_guard = self.input.lock().await;
        // Lazily initialize the InputController on first use (enigo requires
        // Accessibility permission which may be absent at startup time).
        if input_guard.is_none() {
            *input_guard = Some(input::InputController::new().context("InputController init")?);
        }
        let input = input_guard.as_mut().unwrap();
        match args.action {
            Action::Key => {
                let text = args.text.ok_or_else(|| anyhow!("Text required for key"))?;
                input.key_chord(&text)?;
                Ok(ok())
            }
            Action::Type => {
                let text = args.text.ok_or_else(|| anyhow!("Text required for type"))?;
                input.type_text(&text)?;
                Ok(ok())
            }
            Action::MouseMove => {
                let (x, y) = logical_coord
                    .ok_or_else(|| anyhow!("Coordinate required for mouse_move"))?;
                input.mouse_move(x, y)?;
                Ok(ok())
            }
            Action::LeftClick => {
                input.left_click(logical_coord)?;
                Ok(ok())
            }
            Action::LeftClickDrag => {
                let to = logical_coord
                    .ok_or_else(|| anyhow!("Coordinate required for left_click_drag"))?;
                input.left_click_drag(to)?;
                Ok(ok())
            }
            Action::RightClick => {
                input.right_click(logical_coord)?;
                Ok(ok())
            }
            Action::MiddleClick => {
                input.middle_click(logical_coord)?;
                Ok(ok())
            }
            Action::DoubleClick => {
                input.double_click(logical_coord)?;
                Ok(ok())
            }
            Action::Scroll => {
                let at = logical_coord
                    .ok_or_else(|| anyhow!("Coordinate required for scroll"))?;
                let text = args.text.ok_or_else(|| {
                    anyhow!("Text required for scroll (direction like \"up\", \"down:5\")")
                })?;
                let (dir, amt) = input::parse_scroll_text(&text)?;
                input.scroll(at, dir, amt)?;
                Ok(ok())
            }
            Action::GetCursorPosition => {
                let (lx, ly) = input.cursor_position()?;
                let (lw, lh) = self.logical_display_size()?;
                // Convert logical → API image space (inverse of the scaling we apply).
                let scale = 1.0 / scaling::api_to_logical_scale(lw, lh);
                let ax = (lx as f64 * scale).round() as i32;
                let ay = (ly as f64 * scale).round() as i32;
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string(&serde_json::json!({ "x": ax, "y": ay }))
                        .context("serialize cursor position")?,
                )]))
            }
            Action::GetScreenshot => {
                // Release the lock while we sleep so other actions can run.
                // `input` is a &mut ref into `input_guard`; dropping the guard releases the lock.
                let _ = input;
                drop(input_guard);
                // Wait 1 s for animations / loading to settle (matches TS: setTimeout(1000)).
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

                // Re-acquire to read cursor position.
                let mut input_guard2 = self.input.lock().await;
                if input_guard2.is_none() {
                    *input_guard2 = Some(
                        input::InputController::new().context("InputController init")?,
                    );
                }
                let (cx_logical, cy_logical) = input_guard2.as_mut().unwrap().cursor_position()?;
                drop(input_guard2);

                let cap = capture::capture_primary_display()?;

                // Convert logical cursor coords to API image space.
                let scale_logical_to_api =
                    1.0 / scaling::api_to_logical_scale(cap.logical_width, cap.logical_height);
                let cx = (cx_logical as f64 * scale_logical_to_api).round() as i32;
                let cy = (cy_logical as f64 * scale_logical_to_api).round() as i32;

                // Draw crosshair onto the captured (already downsampled) image.
                let mut img =
                    image::load_from_memory(&cap.png_bytes)
                        .context("load captured PNG")?
                        .to_rgba8();
                capture::draw_crosshair(&mut img, cx, cy);

                // Re-encode to PNG.
                let mut png_buf =
                    std::io::Cursor::new(Vec::with_capacity(cap.png_bytes.len()));
                use image::ImageEncoder;
                image::codecs::png::PngEncoder::new(&mut png_buf)
                    .write_image(
                        img.as_raw(),
                        img.width(),
                        img.height(),
                        image::ExtendedColorType::Rgba8,
                    )
                    .context("encode PNG with crosshair")?;

                let b64 = base64_encode(&png_buf.into_inner());
                let img_content = Content::image(b64, "image/png");
                let meta = Content::text(
                    serde_json::to_string(&serde_json::json!({
                        "image_width": cap.reported_width,
                        "image_height": cap.reported_height,
                    }))
                    .context("serialize display meta")?,
                );
                Ok(CallToolResult::success(vec![meta, img_content]))
            }
        }
    }

    fn logical_display_size(&self) -> Result<(u32, u32)> {
        let primary = xcap::Monitor::all()
            .context("xcap::Monitor::all")?
            .into_iter()
            .find(|m| m.is_primary().unwrap_or(false))
            .ok_or_else(|| anyhow!("no primary monitor"))?;
        Ok((primary.width()?, primary.height()?))
    }
}

/// Returns a `{"ok": true}` success result.
fn ok() -> CallToolResult {
    CallToolResult::success(vec![Content::text(
        serde_json::json!({ "ok": true }).to_string(),
    )])
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    STANDARD.encode(bytes)
}

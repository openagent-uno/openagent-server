//! Screen capture and downsampling.
//!
//! Primary path: `xcap` (CGWindow on macOS, X11/Wayland portal on Linux, DXGI on Windows).
//! macOS fallback: shell out to `screencapture -x` for environments where `xcap` regresses
//! (e.g. the CGDisplayCreateImageForRect removal on macOS 26+ that also broke nut-js).

use anyhow::{Context, Result, anyhow};
use fast_image_resize::{Resizer, images::Image as FirImage};
use image::{ImageEncoder, RgbaImage, codecs::png::PngEncoder};
use std::io::Cursor;

use crate::scaling::size_to_api_scale;

pub struct CaptureResult {
    /// PNG-encoded, already downsampled to fit Claude's API limits.
    pub png_bytes: Vec<u8>,
    /// Dimensions AFTER downsampling (what gets reported as display_width_px / display_height_px).
    pub reported_width: u32,
    pub reported_height: u32,
    /// Logical screen dimensions BEFORE downsampling (used for coord scaling).
    pub logical_width: u32,
    pub logical_height: u32,
}

pub fn capture_primary_display() -> Result<CaptureResult> {
    let image = try_xcap().or_else(|e| {
        tracing::warn!("xcap capture failed: {e}, trying fallback");
        try_fallback()
    })?;
    Ok(image)
}

fn try_xcap() -> Result<CaptureResult> {
    let monitors = xcap::Monitor::all().context("xcap::Monitor::all failed")?;
    let primary = monitors
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or_else(|| anyhow!("no primary monitor found"))?;

    let rgba = primary.capture_image().context("xcap capture_image failed")?;
    let logical_w = rgba.width();
    let logical_h = rgba.height();
    // xcap::Monitor::capture_image() returns image::RgbaImage directly
    let rgba: RgbaImage = RgbaImage::from_raw(logical_w, logical_h, rgba.into_raw())
        .ok_or_else(|| anyhow!("xcap returned invalid buffer"))?;
    let (png_bytes, w, h) = downsample_and_encode(rgba)?;
    Ok(CaptureResult {
        png_bytes,
        reported_width: w,
        reported_height: h,
        logical_width: logical_w,
        logical_height: logical_h,
    })
}

#[cfg(target_os = "macos")]
fn try_fallback() -> Result<CaptureResult> {
    use std::process::Command;
    let tmp = std::env::temp_dir().join(format!(
        "openagent-computer-control-{}.png",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let status = Command::new("screencapture")
        .args(["-x", tmp.to_str().ok_or_else(|| anyhow!("non-utf8 tmp path"))?])
        .status()
        .context("spawn screencapture")?;
    if !status.success() {
        return Err(anyhow!("screencapture exited with {status}"));
    }
    let bytes = std::fs::read(&tmp).context("read screencapture png")?;
    let _ = std::fs::remove_file(&tmp);
    let img = image::load_from_memory(&bytes)?.to_rgba8();
    let logical_w = img.width();
    let logical_h = img.height();
    let (png_bytes, w, h) = downsample_and_encode(img)?;
    Ok(CaptureResult {
        png_bytes,
        reported_width: w,
        reported_height: h,
        logical_width: logical_w,
        logical_height: logical_h,
    })
}

#[cfg(not(target_os = "macos"))]
fn try_fallback() -> Result<CaptureResult> {
    Err(anyhow!("no fallback available on this platform"))
}

/// Downsample to fit API limits using Lanczos3. Returns (png_bytes, width, height) in downsampled space.
pub fn downsample_and_encode(src: RgbaImage) -> Result<(Vec<u8>, u32, u32)> {
    let (w, h) = (src.width(), src.height());
    let scale = size_to_api_scale(w, h);
    let out = if (scale - 1.0).abs() < f64::EPSILON {
        src
    } else {
        let new_w = ((w as f64 * scale).floor() as u32).max(1);
        let new_h = ((h as f64 * scale).floor() as u32).max(1);
        // from_vec_u8(width, height, buffer, pixel_type) -> Result<Self, ImageBufferError>
        let src_view = FirImage::from_vec_u8(
            w,
            h,
            src.into_raw(),
            fast_image_resize::PixelType::U8x4,
        )
        .map_err(|e| anyhow!("FirImage::from_vec_u8 failed: {e}"))?;
        let mut dst = FirImage::new(new_w, new_h, fast_image_resize::PixelType::U8x4);
        let mut resizer = Resizer::new();
        resizer
            .resize(&src_view, &mut dst, None)
            .map_err(|e| anyhow!("resize failed: {e}"))?;
        RgbaImage::from_raw(new_w, new_h, dst.into_vec())
            .ok_or_else(|| anyhow!("resize returned invalid buffer"))?
    };
    let (ow, oh) = (out.width(), out.height());
    let mut buf = Cursor::new(Vec::with_capacity((ow * oh * 2) as usize));
    PngEncoder::new(&mut buf)
        .write_image(out.as_raw(), ow, oh, image::ExtendedColorType::Rgba8)?;
    Ok((buf.into_inner(), ow, oh))
}

/// Draw a 20-pixel-half-width red crosshair centered at (cx, cy) in the image,
/// 3 pixels thick. Ports the loop at computer.ts:376-401.
pub fn draw_crosshair(img: &mut RgbaImage, cx: i32, cy: i32) {
    const SIZE: i32 = 20;
    const COLOR: image::Rgba<u8> = image::Rgba([255, 0, 0, 255]);
    let (w, h) = (img.width() as i32, img.height() as i32);
    // Horizontal (3 rows thick for visibility)
    for x in (cx - SIZE).max(0)..=(cx + SIZE).min(w - 1) {
        for dy in [-1, 0, 1] {
            let y = cy + dy;
            if y >= 0 && y < h {
                img.put_pixel(x as u32, y as u32, COLOR);
            }
        }
    }
    // Vertical (3 columns thick)
    for y in (cy - SIZE).max(0)..=(cy + SIZE).min(h - 1) {
        for dx in [-1, 0, 1] {
            let x = cx + dx;
            if x >= 0 && x < w {
                img.put_pixel(x as u32, y as u32, COLOR);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_large_image_shrinks_to_limit() {
        let src = RgbaImage::from_pixel(3840, 2160, image::Rgba([10, 20, 30, 255]));
        let (bytes, w, h) = downsample_and_encode(src).unwrap();
        assert!(w.max(h) <= crate::scaling::MAX_LONG_EDGE);
        assert!((w as u64 * h as u64) as f64 <= crate::scaling::MAX_PIXELS * 1.01);
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn downsample_small_image_unchanged() {
        let src = RgbaImage::from_pixel(800, 600, image::Rgba([1, 2, 3, 255]));
        let (_, w, h) = downsample_and_encode(src).unwrap();
        assert_eq!((w, h), (800, 600));
    }

    #[test]
    fn crosshair_paints_red_at_center() {
        let mut img = RgbaImage::from_pixel(100, 100, image::Rgba([0, 0, 0, 255]));
        draw_crosshair(&mut img, 50, 50);
        assert_eq!(*img.get_pixel(50, 50), image::Rgba([255, 0, 0, 255]));
        assert_eq!(*img.get_pixel(60, 50), image::Rgba([255, 0, 0, 255]));
        assert_eq!(*img.get_pixel(50, 60), image::Rgba([255, 0, 0, 255]));
        assert_eq!(*img.get_pixel(99, 99), image::Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn crosshair_near_edge_does_not_panic() {
        let mut img = RgbaImage::from_pixel(50, 50, image::Rgba([0, 0, 0, 255]));
        draw_crosshair(&mut img, 0, 0);
        draw_crosshair(&mut img, 49, 49);
        draw_crosshair(&mut img, -5, 100);
    }

    #[test]
    #[ignore] // run with `cargo test capture_real -- --ignored --nocapture`
    fn capture_real_display() {
        let r = capture_primary_display().unwrap();
        std::fs::write("/tmp/smoke.png", &r.png_bytes).unwrap();
        println!("logical: {}x{}", r.logical_width, r.logical_height);
        println!("reported: {}x{}", r.reported_width, r.reported_height);
    }
}

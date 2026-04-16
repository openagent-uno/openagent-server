//! Coordinate and image scaling to match Claude's image autoscaling behavior.
//! Claude downsamples images larger than 1568px on the long edge or 1.15MP.
//! We pre-scale so the reported dimensions match what Claude actually sees.

pub const MAX_LONG_EDGE: u32 = 1568;
pub const MAX_PIXELS: f64 = 1.15 * 1024.0 * 1024.0;

/// Scale factor to shrink a (width, height) image to fit API limits. Always `<= 1.0`.
pub fn size_to_api_scale(width: u32, height: u32) -> f64 {
    let long_edge = width.max(height) as f64;
    let total_pixels = (width as u64 * height as u64) as f64;

    let long_edge_scale = if long_edge > MAX_LONG_EDGE as f64 {
        MAX_LONG_EDGE as f64 / long_edge
    } else {
        1.0
    };
    let pixel_scale = if total_pixels > MAX_PIXELS {
        (MAX_PIXELS / total_pixels).sqrt()
    } else {
        1.0
    };
    long_edge_scale.min(pixel_scale)
}

/// Inverse scale: API image coordinates → logical screen coordinates.
pub fn api_to_logical_scale(logical_width: u32, logical_height: u32) -> f64 {
    let api_scale = size_to_api_scale(logical_width, logical_height);
    1.0 / api_scale
}

/// Convert an (x, y) from API image coords to logical screen coords.
pub fn api_to_logical(x: i32, y: i32, logical_w: u32, logical_h: u32) -> (i32, i32) {
    let s = api_to_logical_scale(logical_w, logical_h);
    ((x as f64 * s).round() as i32, (y as f64 * s).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_image_no_scaling() {
        assert_eq!(size_to_api_scale(800, 600), 1.0);
    }

    #[test]
    fn long_edge_scales_down() {
        // Note: the plan's original case used 3840x2160, but pixel-count binds there
        // (pixel scale ≈ 0.381 < long-edge scale ≈ 0.408). 2000x400 has 800k pixels
        // well under MAX_PIXELS, so only the long-edge constraint fires here.
        // 2000x400: long edge 2000 > 1568, but total pixels 800_000 < MAX_PIXELS,
        // so only the long-edge constraint fires → scale = 1568 / 2000.
        let s = size_to_api_scale(2000, 400);
        assert!((s - 1568.0 / 2000.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_count_scales_down() {
        // 1200x1100 = 1_320_000 pixels > 1.15MP (1_205_534). Long edge 1200 < 1568.
        let s = size_to_api_scale(1200, 1100);
        assert!(s < 1.0);
        assert!(s > 0.9);
        let expected = (MAX_PIXELS / (1200.0 * 1100.0)).sqrt();
        assert!((s - expected).abs() < 1e-9);
    }

    #[test]
    fn api_to_logical_applies_inverse_scale() {
        // Claude sends a coord in downsampled space; we scale up to logical.
        // For a 3840x2560 display, pixel-count constraint binds (≈ 0.35023), inverse ≈ 2.8553.
        // (100 * 2.8553).round() = 286.
        let (lx, ly) = api_to_logical(100, 100, 3840, 2560);
        assert_eq!((lx, ly), (286, 286));
    }

    #[test]
    fn common_display_sizes_produce_valid_scales() {
        for (w, h) in [(1920, 1080), (2560, 1440), (3840, 2160), (3840, 2560), (1344, 896)] {
            let s = size_to_api_scale(w, h);
            assert!(s > 0.0 && s <= 1.0, "bad scale {s} for {w}x{h}");
        }
    }
}

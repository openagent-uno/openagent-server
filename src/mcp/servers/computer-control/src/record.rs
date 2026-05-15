//! Screen recording pipeline: capture → H.264 encode → MP4 mux.
//!
//! Pure-Rust stack, no external runtime deps (no ffmpeg, no GStreamer):
//!   * `xcap` supplies per-frame RGBA bitmaps (same backend the screenshot path uses).
//!   * `openh264` encodes YUV420 → H.264 NAL units via Cisco's OpenH264
//!     library (`openh264-sys2` downloads the prebuilt blob at build time,
//!     sidestepping patent licensing the same way Firefox does).
//!   * `mp4` muxes the H.264 stream into a standard `.mp4` file playable in
//!     QuickTime / VLC / Chrome without extra codecs.
//!
//! Frames are captured in a `tokio::task::spawn_blocking` thread so xcap's
//! blocking capture and openh264's CPU-heavy encode stay off the tokio worker
//! pool. A shutdown is signalled via `Arc<AtomicBool>` which the loop polls
//! between frames.
//!
//! Output is downsampled to fit Claude's API image limits (≤1568px long edge,
//! ≤1.15MP) — identical pipeline to the screenshot path in `capture.rs` — so
//! recording resolution matches what the model actually "sees" in screenshots.

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use fast_image_resize::{Resizer, images::Image as FirImage};
use image::RgbaImage;
use mp4::{AvcConfig, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType};
use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::capture::capture_primary_frame_rgba;
use crate::scaling::{LogicalRegion, size_to_api_scale};

/// Minimum and maximum frame rate accepted from MCP callers.
pub const MIN_FPS: u32 = 1;
pub const MAX_FPS: u32 = 60;

/// Active recording session. Created by [`start_recording`]; finalized with
/// [`RecordingSession::stop`].
pub struct RecordingSession {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    stop: Arc<AtomicBool>,
    frames: Arc<AtomicU64>,
    started_at: Instant,
    join: tokio::task::JoinHandle<Result<()>>,
}

/// Outcome returned to the MCP caller when a recording is stopped.
#[derive(Debug, serde::Serialize)]
pub struct RecordingOutcome {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub frames: u64,
    pub duration_seconds: f64,
}

impl RecordingSession {
    /// Signal the capture loop to stop and wait for the MP4 to be finalized.
    pub async fn stop(self) -> Result<RecordingOutcome> {
        self.stop.store(true, Ordering::SeqCst);
        let elapsed = self.started_at.elapsed().as_secs_f64();
        match self.join.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(anyhow!("recording task panicked or was cancelled: {e}")),
        }
        Ok(RecordingOutcome {
            path: self.path.to_string_lossy().into_owned(),
            width: self.width,
            height: self.height,
            fps: self.fps,
            frames: self.frames.load(Ordering::SeqCst),
            duration_seconds: elapsed,
        })
    }

}

/// Start recording the primary display to `path` at `fps` frames/second.
///
/// * `region` — optional crop rectangle in logical screen pixels. Apply the
///   API→logical scaling in the caller (see [`crate::scaling::api_region_to_logical`]).
/// * `max_duration` — optional auto-stop after this many seconds. `None` means
///   record until [`RecordingSession::stop`] is called.
pub fn start_recording(
    path: PathBuf,
    fps: u32,
    region: Option<LogicalRegion>,
    max_duration: Option<Duration>,
) -> Result<RecordingSession> {
    if !(MIN_FPS..=MAX_FPS).contains(&fps) {
        return Err(anyhow!("fps must be between {MIN_FPS} and {MAX_FPS}, got {fps}"));
    }
    if path.exists() {
        return Err(anyhow!(
            "output path already exists: {}",
            path.to_string_lossy()
        ));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(anyhow!(
                "parent directory does not exist: {}",
                parent.to_string_lossy()
            ));
        }
    }

    // Grab one frame upfront to learn the capture dimensions — we need them
    // before we can size the encoder and downsampler. If this fails, the
    // caller sees the error immediately rather than after spawning a task.
    let (probe, _lw, _lh) = capture_primary_frame_rgba(region)
        .context("initial frame capture failed")?;
    let (enc_w, enc_h) = encoder_dims_from_capture(probe.width(), probe.height());
    drop(probe);

    let stop = Arc::new(AtomicBool::new(false));
    let frames = Arc::new(AtomicU64::new(0));
    let stop_task = stop.clone();
    let frames_task = frames.clone();
    let path_task = path.clone();

    let join = tokio::task::spawn_blocking(move || {
        run_capture_loop(
            path_task,
            fps,
            region,
            enc_w,
            enc_h,
            max_duration,
            stop_task,
            frames_task,
        )
    });

    Ok(RecordingSession {
        path,
        width: enc_w,
        height: enc_h,
        fps,
        stop,
        frames,
        started_at: Instant::now(),
        join,
    })
}

/// Compute the encoder's target (width, height) from the captured frame size.
/// Shrinks to fit API limits (same `size_to_api_scale` the screenshot path
/// uses) and forces both dimensions to be even — H.264 YUV420 requires that.
fn encoder_dims_from_capture(cap_w: u32, cap_h: u32) -> (u32, u32) {
    let s = size_to_api_scale(cap_w, cap_h);
    let w = ((cap_w as f64 * s).floor() as u32).max(2);
    let h = ((cap_h as f64 * s).floor() as u32).max(2);
    (w & !1, h & !1)
}

/// The blocking capture+encode+mux loop. Runs on a `spawn_blocking` thread.
#[allow(clippy::too_many_arguments)]
fn run_capture_loop(
    path: PathBuf,
    fps: u32,
    region: Option<LogicalRegion>,
    enc_w: u32,
    enc_h: u32,
    max_duration: Option<Duration>,
    stop: Arc<AtomicBool>,
    frames_counter: Arc<AtomicU64>,
) -> Result<()> {
    let encoder_config = EncoderConfig::new()
        .max_frame_rate(openh264::encoder::FrameRate::from_hz(fps as f32))
        .usage_type(openh264::encoder::UsageType::ScreenContentRealTime);
    let mut encoder = Encoder::with_api_config(
        openh264::OpenH264API::from_source(),
        encoder_config,
    )
    .context("openh264 Encoder init")?;

    let mut resizer = Resizer::new();
    let mut muxer: Option<MuxState<BufWriter<File>>> = None;
    let mut frame_index: u64 = 0;

    let loop_start = Instant::now();
    let frame_interval = Duration::from_secs_f64(1.0 / fps as f64);
    // Timescale = fps so each frame has duration=1 and start_time=frame_index.
    // This gives the MP4 exact integer timestamps with no drift.
    let timescale = fps;

    // Per-frame target deadline, advanced by frame_interval each iteration.
    // Using an absolute deadline avoids drift from per-iteration sleep jitter.
    let mut next_deadline = loop_start;

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if let Some(max) = max_duration {
            if loop_start.elapsed() >= max {
                break;
            }
        }

        // 1. Capture + crop.
        let (rgba, _logical_w, _logical_h) = match capture_primary_frame_rgba(region) {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!("capture failed on frame {frame_index}: {e:#}");
                // Transient capture errors (momentary permission glitches,
                // compositor hiccups) shouldn't kill the whole recording —
                // skip this frame and try the next one.
                pace_to(&mut next_deadline, frame_interval);
                continue;
            }
        };

        // 2. Downsample to encoder resolution (if different).
        let scaled = if rgba.width() == enc_w && rgba.height() == enc_h {
            rgba
        } else {
            resize_rgba(&mut resizer, rgba, enc_w, enc_h)
                .context("resize frame to encoder dims")?
        };

        // 3. RGBA → YUV420 (openh264 input format). `RgbaSliceU8` implements
        // `RGBSource` (the 4-channel variant); use `from_rgb_source` rather
        // than `from_rgb8_source` so the alpha channel is ignored correctly.
        let yuv = YUVBuffer::from_rgb_source(RgbaSliceU8::new(
            scaled.as_raw(),
            (enc_w as usize, enc_h as usize),
        ));

        // 4. Encode.
        let bitstream = encoder.encode(&yuv).context("openh264 encode frame")?;
        let mut annexb_bytes = Vec::with_capacity(64 * 1024);
        bitstream.write_vec(&mut annexb_bytes);
        let is_idr = matches!(
            bitstream.frame_type(),
            openh264::encoder::FrameType::IDR | openh264::encoder::FrameType::I
        );

        // 5. Mux. On the first frame we need to extract SPS/PPS from the
        //    bitstream to build the AVC decoder config; subsequent frames
        //    skip those NAL types since they live in the config box.
        let nals = split_annexb(&annexb_bytes);
        if muxer.is_none() {
            let sps = nals
                .iter()
                .find(|n| nal_type(n) == 7)
                .copied()
                .ok_or_else(|| {
                    anyhow!("first encoded frame contained no SPS NAL unit")
                })?;
            let pps = nals
                .iter()
                .find(|n| nal_type(n) == 8)
                .copied()
                .ok_or_else(|| {
                    anyhow!("first encoded frame contained no PPS NAL unit")
                })?;
            // Open the output file lazily — only once we have an SPS/PPS
            // pair to build the AVC decoder config. If capture or encode
            // fails before this point, we never create a zero-byte file.
            let file = File::create(&path)
                .with_context(|| format!("create {}", path.to_string_lossy()))?;
            muxer = Some(MuxState::new(
                BufWriter::new(file),
                enc_w,
                enc_h,
                timescale,
                sps.to_vec(),
                pps.to_vec(),
            )?);
        }
        let mux = muxer.as_mut().expect("initialized above");

        // Build AVCC sample: drop SPS/PPS/AUD; length-prefix each remaining NAL.
        let mut avcc = Vec::with_capacity(annexb_bytes.len());
        for nal in &nals {
            let t = nal_type(nal);
            if t == 7 || t == 8 || t == 9 {
                // SPS, PPS, Access-Unit-Delimiter: skipped.
                continue;
            }
            let len = nal.len() as u32;
            avcc.extend_from_slice(&len.to_be_bytes());
            avcc.extend_from_slice(nal);
        }
        if avcc.is_empty() {
            // Pure-parameter-set frame (shouldn't happen in normal flow, but
            // don't write zero-byte samples if it does).
            pace_to(&mut next_deadline, frame_interval);
            continue;
        }

        mux.writer.write_sample(
            1,
            &Mp4Sample {
                start_time: frame_index,
                duration: 1,
                rendering_offset: 0,
                is_sync: is_idr,
                bytes: Bytes::from(avcc),
            },
        )
        .with_context(|| format!("mp4 write_sample frame {frame_index}"))?;

        frame_index += 1;
        frames_counter.store(frame_index, Ordering::SeqCst);
        pace_to(&mut next_deadline, frame_interval);
    }

    if let Some(mut mux) = muxer {
        mux.writer.write_end().context("mp4 write_end")?;
        mux.writer
            .into_writer()
            .flush()
            .context("flush mp4 output")?;
        Ok(())
    } else {
        // Stopped before any frame made it through the encoder. The output
        // file was never created (we open it lazily), so there's nothing
        // to clean up on disk.
        Err(anyhow!(
            "recording stopped before any frames were captured"
        ))
    }
}

/// Sleep until `*deadline`, then advance it by `interval`.
/// Uses absolute deadlines so pacing doesn't drift.
fn pace_to(deadline: &mut Instant, interval: Duration) {
    *deadline += interval;
    let now = Instant::now();
    if *deadline > now {
        std::thread::sleep(*deadline - now);
    } else {
        // We're behind schedule (encode took longer than the frame budget).
        // Skip ahead rather than accumulating debt; the effective fps drops
        // but we don't end up rendering future frames back-to-back.
        *deadline = now;
    }
}

fn resize_rgba(
    resizer: &mut Resizer,
    src: RgbaImage,
    out_w: u32,
    out_h: u32,
) -> Result<RgbaImage> {
    let (w, h) = (src.width(), src.height());
    let src_view = FirImage::from_vec_u8(
        w,
        h,
        src.into_raw(),
        fast_image_resize::PixelType::U8x4,
    )
    .map_err(|e| anyhow!("FirImage::from_vec_u8: {e}"))?;
    let mut dst = FirImage::new(out_w, out_h, fast_image_resize::PixelType::U8x4);
    resizer
        .resize(&src_view, &mut dst, None)
        .map_err(|e| anyhow!("fast_image_resize: {e}"))?;
    RgbaImage::from_raw(out_w, out_h, dst.into_vec())
        .ok_or_else(|| anyhow!("resize returned invalid buffer"))
}

struct MuxState<W: Write + Seek> {
    writer: Mp4Writer<W>,
}

impl<W: Write + Seek> MuxState<W> {
    fn new(
        w: W,
        width: u32,
        height: u32,
        timescale: u32,
        sps: Vec<u8>,
        pps: Vec<u8>,
    ) -> Result<Self> {
        let cfg = Mp4Config {
            major_brand: "isom".parse().map_err(|e| anyhow!("parse brand: {e}"))?,
            minor_version: 512,
            compatible_brands: vec![
                "isom".parse().map_err(|e| anyhow!("parse brand: {e}"))?,
                "iso2".parse().map_err(|e| anyhow!("parse brand: {e}"))?,
                "avc1".parse().map_err(|e| anyhow!("parse brand: {e}"))?,
                "mp41".parse().map_err(|e| anyhow!("parse brand: {e}"))?,
            ],
            timescale,
        };
        let mut writer = Mp4Writer::write_start(w, &cfg).context("mp4 write_start")?;
        let track = TrackConfig {
            track_type: TrackType::Video,
            timescale,
            language: "und".to_string(),
            media_conf: MediaConfig::AvcConfig(AvcConfig {
                width: u16::try_from(width).unwrap_or(u16::MAX),
                height: u16::try_from(height).unwrap_or(u16::MAX),
                seq_param_set: sps,
                pic_param_set: pps,
            }),
        };
        writer.add_track(&track).context("mp4 add_track")?;
        Ok(Self { writer })
    }
}

/// Split an Annex-B H.264 bitstream into its constituent NAL unit payloads
/// (without the start codes).
///
/// Start codes are `0x00 0x00 0x00 0x01` or `0x00 0x00 0x01`. We do a simple
/// linear scan — at the bitrates/sizes we encode at, this is fine.
fn split_annexb(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut nal_start: Option<usize> = None;
    while i < buf.len() {
        let (match_len, found) = if i + 4 <= buf.len()
            && buf[i] == 0
            && buf[i + 1] == 0
            && buf[i + 2] == 0
            && buf[i + 3] == 1
        {
            (4, true)
        } else if i + 3 <= buf.len() && buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 {
            (3, true)
        } else {
            (0, false)
        };
        if found {
            if let Some(start) = nal_start {
                // The previous NAL ends just before this start code.
                out.push(&buf[start..i]);
            }
            nal_start = Some(i + match_len);
            i += match_len;
        } else {
            i += 1;
        }
    }
    if let Some(start) = nal_start {
        if start < buf.len() {
            out.push(&buf[start..]);
        }
    }
    out
}

/// Return the H.264 NAL unit type (low 5 bits of the first byte).
fn nal_type(nal: &[u8]) -> u8 {
    if nal.is_empty() { 0 } else { nal[0] & 0x1F }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_dims_rounds_to_even() {
        // Arbitrary odd-dimension capture should produce even encoder dims.
        let (w, h) = encoder_dims_from_capture(1001, 701);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
        assert!(w > 0 && h > 0);
    }

    #[test]
    fn encoder_dims_honors_api_limits() {
        let (w, h) = encoder_dims_from_capture(3840, 2160);
        assert!(w.max(h) <= crate::scaling::MAX_LONG_EDGE);
        assert!((w as u64 * h as u64) as f64 <= crate::scaling::MAX_PIXELS * 1.01);
    }

    #[test]
    fn split_annexb_handles_3_and_4_byte_start_codes() {
        // Two NALs: first uses 4-byte start code, second uses 3-byte.
        let buf: &[u8] = &[
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f, // SPS (type 7)
            0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, // PPS (type 8)
        ];
        let nals = split_annexb(buf);
        assert_eq!(nals.len(), 2);
        assert_eq!(nal_type(nals[0]), 7);
        assert_eq!(nal_type(nals[1]), 8);
    }

    #[test]
    fn nal_type_extracts_low_5_bits() {
        assert_eq!(nal_type(&[0x67]), 7);
        assert_eq!(nal_type(&[0x68]), 8);
        assert_eq!(nal_type(&[0x65]), 5); // IDR slice
        assert_eq!(nal_type(&[]), 0);
    }

    #[test]
    fn pace_to_advances_deadline() {
        let mut d = Instant::now() - Duration::from_millis(100);
        let before = d;
        pace_to(&mut d, Duration::from_millis(33));
        // When we're behind, deadline is reset to "now" — strictly later than
        // the initial past-deadline.
        assert!(d >= before);
    }

    #[test]
    #[ignore] // run on a real machine with a display:
    //   cargo test record_real -- --ignored --nocapture
    fn record_real_display() {
        // Smoke test: record ~2s at 30 fps and print the output path.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let path = std::env::temp_dir().join("openagent-rec-smoke.mp4");
            let _ = std::fs::remove_file(&path);
            let session = start_recording(
                path.clone(),
                30,
                None,
                Some(Duration::from_secs(2)),
            )
            .unwrap();
            tokio::time::sleep(Duration::from_millis(2200)).await;
            let outcome = session.stop().await.unwrap();
            println!(
                "recorded {} frames, {:.2}s, {}x{} → {}",
                outcome.frames,
                outcome.duration_seconds,
                outcome.width,
                outcome.height,
                outcome.path,
            );
            assert!(outcome.frames > 0);
            let meta = std::fs::metadata(&path).unwrap();
            assert!(meta.len() > 0);
        });
    }
}

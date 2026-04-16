//! Parse xdotool-style key strings (e.g. "ctrl+shift+a", "Return", "super+space")
//! into enigo key presses. Mirrors ../src/xdotoolStringToKeys.ts from the
//! previous TypeScript implementation.
//!
//! Mapping notes vs the TypeScript source (nut-js Key enum → enigo Key enum):
//! - nut-js `LeftSuper`/`RightSuper` → enigo `Meta` (the one cross-platform super/cmd/win key).
//! - nut-js letters (Key.A…Key.Z) and digits (Key.Num0…Key.Num9) are mapped as
//!   `Key::Unicode(c)` because enigo's named letter/digit variants are
//!   `#[cfg(target_os = "windows")]`-only. Using Unicode works on all platforms.
//! - nut-js punctuation variants (Key.Minus, Key.Equal, Key.Semicolon, …) → `Key::Unicode(c)`.
//!   enigo has no dedicated punctuation key variants; Unicode fallback is correct.
//! - nut-js `NumPadEqual` → no equivalent in enigo 0.5; skipped with a comment below.
//! - nut-js `AudioMute`/`AudioVolDown`/`AudioVolUp` → enigo `VolumeMute`/`VolumeDown`/`VolumeUp`.
//! - nut-js `AudioPlay`/`AudioPause` → enigo `MediaPlayPause`.
//! - nut-js `AudioStop` → enigo `MediaStop` (linux/windows only; on macOS emits `Other(0)`
//!   which is harmless but a no-op).
//! - nut-js `AudioPrev`/`AudioNext` → enigo `MediaPrevTrack`/`MediaNextTrack`.
//! - enigo `Insert` is `#[cfg(any(windows, linux))]`; on macOS it resolves to `Other(0)`.
//! - enigo `Numlock` is `#[cfg(any(windows, linux))]`; same note as Insert.

use enigo::Key;

/// Parse a xdotool-style key spec into an ordered list of keys to press together.
/// Example: `"ctrl+shift+a"` → `[Key::LControl, Key::LShift, Key::Unicode('a')]`.
pub fn parse(spec: &str) -> Result<Vec<Key>, String> {
    spec.split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(parse_single)
        .collect()
}

fn parse_single(name: &str) -> Result<Key, String> {
    let lower = name.to_ascii_lowercase();
    if let Some(k) = lookup(&lower) {
        return Ok(k);
    }
    // Single-character fallback (alphanumerics not in the explicit map).
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        return Ok(Key::Unicode(c));
    }
    Err(format!("Unknown key: {name}"))
}

fn lookup(name: &str) -> Option<Key> {
    Some(match name {
        // ── Function keys ────────────────────────────────────────────────────────
        "f1"  => Key::F1,
        "f2"  => Key::F2,
        "f3"  => Key::F3,
        "f4"  => Key::F4,
        "f5"  => Key::F5,
        "f6"  => Key::F6,
        "f7"  => Key::F7,
        "f8"  => Key::F8,
        "f9"  => Key::F9,
        "f10" => Key::F10,
        "f11" => Key::F11,
        "f12" => Key::F12,
        // F13–F20 are available on all platforms in enigo 0.5.
        "f13" => Key::F13,
        "f14" => Key::F14,
        "f15" => Key::F15,
        "f16" => Key::F16,
        "f17" => Key::F17,
        "f18" => Key::F18,
        "f19" => Key::F19,
        "f20" => Key::F20,
        // F21–F24 are windows/linux only in enigo; on macOS they compile away.
        // The TS source maps them too, so we keep them — a macOS caller that
        // tries these will get a compile error on that platform (no-op on
        // others). We guard with cfg so the module still compiles everywhere.
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "f21" => Key::F21,
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "f22" => Key::F22,
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "f23" => Key::F23,
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "f24" => Key::F24,

        // ── Navigation ───────────────────────────────────────────────────────────
        "home"                          => Key::Home,
        "left"                          => Key::LeftArrow,
        "up"                            => Key::UpArrow,
        "right"                         => Key::RightArrow,
        "down"                          => Key::DownArrow,
        "page_up" | "pageup" | "prior"  => Key::PageUp,
        "page_down" | "pagedown" | "next" => Key::PageDown,
        "end"                           => Key::End,

        // ── Editing ──────────────────────────────────────────────────────────────
        "return" | "enter"   => Key::Return,
        "tab"                => Key::Tab,
        "space"              => Key::Space,
        "backspace"          => Key::Backspace,
        "delete" | "del"     => Key::Delete,
        "escape" | "esc"     => Key::Escape,
        // Insert is windows/linux only in enigo; macOS ignores it.
        // We keep the mapping so the parser accepts the key name everywhere.
        // Insert is not available on macOS in enigo 0.5; use Other(0) as a no-op sentinel there.
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "insert" | "ins"     => Key::Insert,
        #[cfg(target_os = "macos")]
        "insert" | "ins"     => Key::Other(0),

        // ── Modifiers ────────────────────────────────────────────────────────────
        // Shift
        "shift" | "shift_l" | "l_shift"      => Key::LShift,
        "shift_r" | "r_shift"                 => Key::RShift,

        // Control
        "control" | "ctrl" | "control_l" | "ctrl_l" | "l_ctrl" | "l_control"
            => Key::LControl,
        "control_r" | "ctrl_r" | "r_ctrl" | "r_control"
            => Key::RControl,

        // Alt  (enigo has a single Alt + Option alias; no separate LControl/RAlt variants)
        "alt" | "alt_l" | "l_alt"            => Key::Alt,
        "alt_r" | "r_alt"                     => Key::Alt,

        // Super / Win / Meta / Cmd  — enigo's Meta is the one cross-platform
        // "windows/super/command" key.  nut-js had LeftSuper/RightSuper which
        // enigo collapses into Meta.
        "super" | "super_l" | "l_super"
            | "win" | "win_l" | "l_win"
            | "meta" | "meta_l" | "l_meta"
            | "command" | "command_l" | "l_command"
            | "cmd" | "cmd_l" | "l_cmd"          => Key::Meta,
        "super_r" | "r_super"
            | "win_r" | "r_win"
            | "meta_r" | "r_meta"
            | "command_r" | "r_command"
            | "cmd_r" | "r_cmd"                   => Key::Meta,

        // Caps lock
        "caps_lock" | "capslock" | "caps"    => Key::CapsLock,

        // ── Keypad ───────────────────────────────────────────────────────────────
        "kp_0"        => Key::Numpad0,
        "kp_1"        => Key::Numpad1,
        "kp_2"        => Key::Numpad2,
        "kp_3"        => Key::Numpad3,
        "kp_4"        => Key::Numpad4,
        "kp_5"        => Key::Numpad5,
        "kp_6"        => Key::Numpad6,
        "kp_7"        => Key::Numpad7,
        "kp_8"        => Key::Numpad8,
        "kp_9"        => Key::Numpad9,
        "kp_divide"   => Key::Divide,
        "kp_multiply" => Key::Multiply,
        "kp_subtract" => Key::Subtract,
        "kp_add"      => Key::Add,
        "kp_decimal"  => Key::Decimal,
        // nut-js Key.NumPadEqual has no equivalent in enigo 0.5; skip.
        // "kp_equal" => <no enigo variant>,

        // Num lock (windows/linux only in enigo)
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "num_lock" | "numlock"            => Key::Numlock,
        #[cfg(target_os = "macos")]
        "num_lock" | "numlock"            => Key::Other(0),

        // ── Letters (a–z) and digits (0–9) ──────────────────────────────────────
        // nut-js had dedicated Key.A … Key.Z, Key.Num0 … Key.Num9.
        // enigo's named letter/digit variants are windows-only.
        //
        // We intentionally do NOT handle single-letter/digit keys here.
        // The `parse_single` fallback sends them as Key::Unicode preserving the
        // original case (e.g. "Z" → Unicode('Z'), "a" → Unicode('a')).  Putting
        // them in the explicit map would force lowercase because `lookup` receives
        // a lowercased string, breaking "Z" → Unicode('Z').

        // ── Punctuation ──────────────────────────────────────────────────────────
        // nut-js had dedicated variants (Key.Minus, Key.Equal, …); enigo has none.
        "minus"                          => Key::Unicode('-'),
        "equal"                          => Key::Unicode('='),
        "bracketleft" | "bracket_l" | "l_bracket"  => Key::Unicode('['),
        "bracketright" | "bracket_r" | "r_bracket" => Key::Unicode(']'),
        "backslash"                      => Key::Unicode('\\'),
        "semicolon" | "semi"             => Key::Unicode(';'),
        "quote"                          => Key::Unicode('\''),
        "grave"                          => Key::Unicode('`'),
        "comma"                          => Key::Unicode(','),
        "period"                         => Key::Unicode('.'),
        "slash"                          => Key::Unicode('/'),

        // ── Media keys ───────────────────────────────────────────────────────────
        "audio_mute" | "mute"                        => Key::VolumeMute,
        "audio_vol_down" | "vol_down" | "voldown"    => Key::VolumeDown,
        "audio_vol_up" | "vol_up" | "volup"          => Key::VolumeUp,
        // nut-js AudioPlay and AudioPause both map to enigo's toggle MediaPlayPause.
        "audio_play" | "play" | "audio_pause" | "pause" => Key::MediaPlayPause,
        // MediaStop is linux/windows only in enigo (no macOS variant); Other(0) on macOS.
        #[cfg(any(target_os = "windows", all(unix, not(target_os = "macos"))))]
        "audio_stop" | "stop"                        => Key::MediaStop,
        #[cfg(target_os = "macos")]
        "audio_stop" | "stop"                        => Key::Other(0),
        "audio_prev"                                 => Key::MediaPrevTrack,
        "audio_next"                                 => Key::MediaNextTrack,

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_letter() {
        assert_eq!(parse("a").unwrap(), vec![Key::Unicode('a')]);
        assert_eq!(parse("Z").unwrap(), vec![Key::Unicode('Z')]);
    }

    #[test]
    fn named_keys() {
        assert_eq!(parse("Return").unwrap(), vec![Key::Return]);
        assert_eq!(parse("escape").unwrap(), vec![Key::Escape]);
        assert_eq!(parse("space").unwrap(), vec![Key::Space]);
    }

    #[test]
    fn chord() {
        assert_eq!(
            parse("ctrl+a").unwrap(),
            vec![Key::LControl, Key::Unicode('a')]
        );
        assert_eq!(
            parse("ctrl+shift+t").unwrap(),
            vec![Key::LControl, Key::LShift, Key::Unicode('t')]
        );
    }

    #[test]
    fn aliases_equal() {
        assert_eq!(parse("cmd+q").unwrap(), parse("super+q").unwrap());
        assert_eq!(parse("enter").unwrap(), parse("return").unwrap());
    }

    #[test]
    fn unknown_key_errors() {
        assert!(parse("bogus_key").is_err());
    }

    #[test]
    fn whitespace_in_chord() {
        assert_eq!(
            parse("ctrl + a").unwrap(),
            vec![Key::LControl, Key::Unicode('a')]
        );
    }

    #[test]
    fn function_keys() {
        assert_eq!(parse("F1").unwrap(), vec![Key::F1]);
        assert_eq!(parse("f12").unwrap(), vec![Key::F12]);
        assert_eq!(parse("f13").unwrap(), vec![Key::F13]);
        assert_eq!(parse("f20").unwrap(), vec![Key::F20]);
    }

    #[test]
    fn navigation_keys() {
        assert_eq!(parse("home").unwrap(), vec![Key::Home]);
        assert_eq!(parse("end").unwrap(), vec![Key::End]);
        assert_eq!(parse("left").unwrap(), vec![Key::LeftArrow]);
        assert_eq!(parse("right").unwrap(), vec![Key::RightArrow]);
        assert_eq!(parse("up").unwrap(), vec![Key::UpArrow]);
        assert_eq!(parse("down").unwrap(), vec![Key::DownArrow]);
        assert_eq!(parse("page_up").unwrap(), vec![Key::PageUp]);
        assert_eq!(parse("pageup").unwrap(), vec![Key::PageUp]);
        assert_eq!(parse("prior").unwrap(), vec![Key::PageUp]);
        assert_eq!(parse("page_down").unwrap(), vec![Key::PageDown]);
        assert_eq!(parse("pagedown").unwrap(), vec![Key::PageDown]);
        assert_eq!(parse("next").unwrap(), vec![Key::PageDown]);
    }

    #[test]
    fn modifier_aliases() {
        // Left-hand variants
        assert_eq!(parse("shift_l").unwrap(), parse("l_shift").unwrap());
        assert_eq!(parse("ctrl").unwrap(), parse("control").unwrap());
        assert_eq!(parse("ctrl_l").unwrap(), parse("l_ctrl").unwrap());
        assert_eq!(parse("control_l").unwrap(), parse("l_control").unwrap());
        // Right-hand variants
        assert_eq!(parse("shift_r").unwrap(), vec![Key::RShift]);
        assert_eq!(parse("control_r").unwrap(), vec![Key::RControl]);
        assert_eq!(parse("ctrl_r").unwrap(), vec![Key::RControl]);
        assert_eq!(parse("r_ctrl").unwrap(), vec![Key::RControl]);
        // super/win/meta/cmd all resolve to Meta
        assert_eq!(parse("win").unwrap(), parse("super").unwrap());
        assert_eq!(parse("meta").unwrap(), parse("super").unwrap());
        assert_eq!(parse("command").unwrap(), parse("super").unwrap());
        assert_eq!(parse("cmd").unwrap(), parse("super").unwrap());
        // caps lock aliases
        assert_eq!(parse("capslock").unwrap(), vec![Key::CapsLock]);
        assert_eq!(parse("caps").unwrap(), vec![Key::CapsLock]);
    }

    #[test]
    fn numpad_keys() {
        assert_eq!(parse("kp_0").unwrap(), vec![Key::Numpad0]);
        assert_eq!(parse("kp_9").unwrap(), vec![Key::Numpad9]);
        assert_eq!(parse("kp_divide").unwrap(), vec![Key::Divide]);
        assert_eq!(parse("kp_multiply").unwrap(), vec![Key::Multiply]);
        assert_eq!(parse("kp_subtract").unwrap(), vec![Key::Subtract]);
        assert_eq!(parse("kp_add").unwrap(), vec![Key::Add]);
        assert_eq!(parse("kp_decimal").unwrap(), vec![Key::Decimal]);
    }

    #[test]
    fn digit_keys() {
        assert_eq!(parse("0").unwrap(), vec![Key::Unicode('0')]);
        assert_eq!(parse("9").unwrap(), vec![Key::Unicode('9')]);
    }

    #[test]
    fn punctuation_keys() {
        assert_eq!(parse("minus").unwrap(), vec![Key::Unicode('-')]);
        assert_eq!(parse("equal").unwrap(), vec![Key::Unicode('=')]);
        assert_eq!(parse("bracketleft").unwrap(), vec![Key::Unicode('[')]);
        assert_eq!(parse("bracket_l").unwrap(), vec![Key::Unicode('[')]);
        assert_eq!(parse("bracketright").unwrap(), vec![Key::Unicode(']')]);
        assert_eq!(parse("bracket_r").unwrap(), vec![Key::Unicode(']')]);
        assert_eq!(parse("backslash").unwrap(), vec![Key::Unicode('\\')]);
        assert_eq!(parse("semicolon").unwrap(), vec![Key::Unicode(';')]);
        assert_eq!(parse("semi").unwrap(), vec![Key::Unicode(';')]);
        assert_eq!(parse("quote").unwrap(), vec![Key::Unicode('\'')]);
        assert_eq!(parse("grave").unwrap(), vec![Key::Unicode('`')]);
        assert_eq!(parse("comma").unwrap(), vec![Key::Unicode(',')]);
        assert_eq!(parse("period").unwrap(), vec![Key::Unicode('.')]);
        assert_eq!(parse("slash").unwrap(), vec![Key::Unicode('/')]);
    }

    #[test]
    fn media_keys() {
        assert_eq!(parse("mute").unwrap(), vec![Key::VolumeMute]);
        assert_eq!(parse("audio_mute").unwrap(), vec![Key::VolumeMute]);
        assert_eq!(parse("voldown").unwrap(), vec![Key::VolumeDown]);
        assert_eq!(parse("vol_down").unwrap(), vec![Key::VolumeDown]);
        assert_eq!(parse("audio_vol_down").unwrap(), vec![Key::VolumeDown]);
        assert_eq!(parse("volup").unwrap(), vec![Key::VolumeUp]);
        assert_eq!(parse("vol_up").unwrap(), vec![Key::VolumeUp]);
        assert_eq!(parse("audio_vol_up").unwrap(), vec![Key::VolumeUp]);
        assert_eq!(parse("play").unwrap(), vec![Key::MediaPlayPause]);
        assert_eq!(parse("audio_play").unwrap(), vec![Key::MediaPlayPause]);
        assert_eq!(parse("pause").unwrap(), vec![Key::MediaPlayPause]);
        assert_eq!(parse("audio_pause").unwrap(), vec![Key::MediaPlayPause]);
        assert_eq!(parse("audio_prev").unwrap(), vec![Key::MediaPrevTrack]);
        assert_eq!(parse("audio_next").unwrap(), vec![Key::MediaNextTrack]);
    }
}

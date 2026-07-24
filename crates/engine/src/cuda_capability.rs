//! Shared validation for CUDA compute-capability identities.

pub(crate) const fn ptx_target_matches_compute_capability(
    target: &str,
    expected_major: i32,
    expected_minor: i32,
) -> bool {
    let bytes = target.as_bytes();
    if bytes.len() < 5 || bytes[0] != b's' || bytes[1] != b'm' || bytes[2] != b'_' {
        return false;
    }
    let final_index = bytes.len() - 1;
    let minor_byte = bytes[final_index];
    let Some(minor) = ascii_decimal_digit_value(minor_byte) else {
        return false;
    };
    let mut major = 0_i32;
    let mut index = 3;
    while index < final_index {
        let byte = bytes[index];
        let Some(digit) = ascii_decimal_digit_value(byte) else {
            return false;
        };
        if major > (i32::MAX - digit) / 10 {
            return false;
        }
        major = major * 10 + digit;
        index += 1;
    }
    major == expected_major && minor == expected_minor
}

const fn ascii_decimal_digit_value(byte: u8) -> Option<i32> {
    match byte {
        b'0' => Some(0),
        b'1' => Some(1),
        b'2' => Some(2),
        b'3' => Some(3),
        b'4' => Some(4),
        b'5' => Some(5),
        b'6' => Some(6),
        b'7' => Some(7),
        b'8' => Some(8),
        b'9' => Some(9),
        _ => None,
    }
}

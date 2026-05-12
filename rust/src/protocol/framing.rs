use std::io::{Read, Write};
use std::net::TcpStream;

use crate::protocol::ProtocolError;

/// 【帧长度字段的字节数】每个消息帧前面有 4 字节表示 payload 长度。
pub(crate) const FRAME_LEN_BYTES: usize = 4;
/// 【最大帧大小】16MB，防止恶意/损坏数据导致内存爆炸。
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn write_frame_to_stream(stream: &mut TcpStream, frame: &[u8]) -> Result<usize, ProtocolError> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame.len() });
    }
    let frame_len = u32::try_from(frame.len())
        .map_err(|_| ProtocolError::FrameTooLarge { bytes: frame.len() })?;
    stream.write_all(&frame_len.to_be_bytes())?;
    stream.write_all(frame)?;
    Ok(FRAME_LEN_BYTES + frame.len())
}

pub(crate) fn read_frame_from_stream(stream: &mut TcpStream) -> Result<Vec<u8>, ProtocolError> {
    let mut len_bytes = [0_u8; FRAME_LEN_BYTES];
    stream.read_exact(&mut len_bytes)?;
    let frame_len = u32::from_be_bytes(len_bytes) as usize;
    if frame_len > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge { bytes: frame_len });
    }
    let mut frame = vec![0_u8; frame_len];
    stream.read_exact(&mut frame)?;
    Ok(frame)
}

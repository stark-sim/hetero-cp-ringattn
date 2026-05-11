use std::net::TcpStream;
use std::sync::mpsc::{Receiver, Sender};

use crate::protocol::{ProtocolError, RingAttnMessage};
use crate::protocol::framing::{read_frame_from_stream, write_frame_to_stream, FRAME_LEN_BYTES};

/// 发送端：负责把 RingAttnMessage 序列化后发出，返回实际发出的字节数。
pub trait MessageSender {
    fn send_message(&mut self, message: &RingAttnMessage) -> Result<usize, ProtocolError>;
}

/// 接收端：负责接收 bytes 并反序列化为 RingAttnMessage，返回 (message, bytes_received)。
pub trait MessageReceiver {
    fn recv_message(&mut self) -> Result<(RingAttnMessage, usize), ProtocolError>;
}

impl MessageSender for TcpStream {
    fn send_message(&mut self, message: &RingAttnMessage) -> Result<usize, ProtocolError> {
        let frame = crate::protocol::message::serialize_message(message)?;
        write_frame_to_stream(self, &frame)
    }
}

impl MessageReceiver for TcpStream {
    fn recv_message(&mut self) -> Result<(RingAttnMessage, usize), ProtocolError> {
        let frame = read_frame_from_stream(self)?;
        let message = crate::protocol::message::deserialize_message(&frame)?;
        Ok((message, FRAME_LEN_BYTES + frame.len()))
    }
}

impl MessageSender for Sender<Vec<u8>> {
    fn send_message(&mut self, message: &RingAttnMessage) -> Result<usize, ProtocolError> {
        let frame = crate::protocol::message::serialize_message(message)?;
        let len = frame.len();
        self.send(frame).map_err(|error| ProtocolError::ChannelSend {
            sender_domain: message.sender_domain.clone(),
            receiver_domain: message.receiver_domain.clone(),
            reason: error.to_string(),
        })?;
        Ok(len)
    }
}

impl MessageReceiver for Receiver<Vec<u8>> {
    fn recv_message(&mut self) -> Result<(RingAttnMessage, usize), ProtocolError> {
        let frame = self.recv().map_err(|error| ProtocolError::ChannelRecv {
            domain_id: "channel".to_string(),
            reason: error.to_string(),
        })?;
        let len = frame.len();
        let message = crate::protocol::message::deserialize_message(&frame)?;
        Ok((message, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::message::sample_kv_message;

    /// 验证 MessageSender / MessageReceiver trait 通过本地 TCP 端到端工作正常。
    #[test]
    fn test_message_sender_receiver_tcp_roundtrip() {
        use std::net::{TcpListener, TcpStream};
        use std::thread;

        let original = sample_kv_message();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let expected = original.clone();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut receiver: TcpStream = stream;
            let (msg, _bytes) = MessageReceiver::recv_message(&mut receiver).unwrap();
            msg
        });

        let client = thread::spawn(move || {
            let stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            let mut sender: TcpStream = stream;
            MessageSender::send_message(&mut sender, &expected).unwrap();
        });

        client.join().unwrap();
        let received = server.join().unwrap();

        assert_eq!(original, received);
    }
}

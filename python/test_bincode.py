"""Unit tests for Python bincode encoder/decoder against Rust ground truth."""
import unittest
import sys
sys.path.insert(0, "python")

from hcp_worker_sdk.bincode import (
    encode_command, decode_command,
    encode_response, decode_response,
    encode_handshake, decode_handshake,
)


class TestBincode(unittest.TestCase):

    def test_prefill_cmd(self):
        """Match Rust: WorkerCommand::Prefill { chunk: vec![1,2,3], seq_offset: 0 }"""
        data = encode_command("Prefill", chunk=[1, 2, 3], seq_offset=0)
        expected = bytes([
            0, 0, 0, 0,          # tag = 0
            3, 0, 0, 0, 0, 0, 0, 0,  # vec len = 3
            1, 0, 0, 0, 0, 0, 0, 0,  # chunk[0] = 1
            2, 0, 0, 0, 0, 0, 0, 0,  # chunk[1] = 2
            3, 0, 0, 0, 0, 0, 0, 0,  # chunk[2] = 3
            0, 0, 0, 0, 0, 0, 0, 0,  # seq_offset = 0
        ])
        self.assertEqual(data, expected)
        decoded = decode_command(data)
        self.assertEqual(decoded["kind"], "Prefill")
        self.assertEqual(decoded["chunk"], [1, 2, 3])
        self.assertEqual(decoded["seq_offset"], 0)

    def test_decode_cmd(self):
        """Match Rust: WorkerCommand::Decode(42)"""
        data = encode_command("Decode", token=42)
        expected = bytes([
            1, 0, 0, 0,          # tag = 1
            42, 0, 0, 0, 0, 0, 0, 0,  # token = 42
        ])
        self.assertEqual(data, expected)
        decoded = decode_command(data)
        self.assertEqual(decoded["kind"], "Decode")
        self.assertEqual(decoded["token"], 42)

    def test_sync_global_seq_len_cmd(self):
        """Match Rust: WorkerCommand::SyncGlobalSeqLen(11)"""
        data = encode_command("SyncGlobalSeqLen", global_seq_len=11)
        expected = bytes([
            2, 0, 0, 0,          # tag = 2
            11, 0, 0, 0, 0, 0, 0, 0,  # global_seq_len = 11
        ])
        self.assertEqual(data, expected)
        decoded = decode_command(data)
        self.assertEqual(decoded["kind"], "SyncGlobalSeqLen")
        self.assertEqual(decoded["global_seq_len"], 11)

    def test_shutdown_cmd(self):
        """Match Rust: WorkerCommand::Shutdown"""
        data = encode_command("Shutdown")
        expected = bytes([
            3, 0, 0, 0,          # tag = 3
        ])
        self.assertEqual(data, expected)
        decoded = decode_command(data)
        self.assertEqual(decoded["kind"], "Shutdown")

    def test_prefill_done_resp(self):
        """Match Rust: WorkerResponse::PrefillDone { last_logits_bytes: vec![0xAB, 0xCD], global_seq_len: 11 }"""
        data = encode_response("PrefillDone", last_logits_bytes=bytes([0xAB, 0xCD]), global_seq_len=11)
        expected = bytes([
            0, 0, 0, 0,          # tag = 0
            2, 0, 0, 0, 0, 0, 0, 0,  # bytes len = 2
            0xAB, 0xCD,          # bytes
            11, 0, 0, 0, 0, 0, 0, 0,  # global_seq_len = 11
        ])
        self.assertEqual(data, expected)
        decoded = decode_response(data)
        self.assertEqual(decoded["kind"], "PrefillDone")
        self.assertEqual(decoded["last_logits_bytes"], bytes([0xAB, 0xCD]))
        self.assertEqual(decoded["global_seq_len"], 11)

    def test_handshake(self):
        """Match Rust: WorkerHandshake { domain_id: 0, capacity_mb: 4096 }"""
        data = encode_handshake(0, 4096)
        expected = bytes([
            0, 0, 0, 0, 0, 0, 0, 0,  # domain_id = 0
            0, 16, 0, 0, 0, 0, 0, 0,  # capacity_mb = 4096
        ])
        self.assertEqual(data, expected)
        domain_id, capacity_mb = decode_handshake(data)
        self.assertEqual(domain_id, 0)
        self.assertEqual(capacity_mb, 4096)


if __name__ == "__main__":
    unittest.main(verbosity=2)

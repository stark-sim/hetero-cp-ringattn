fn main() -> Result<(), String> {
    hcp_ringattn_rust::run_self_driving_quic_smoke(&std::env::args().skip(1).collect::<Vec<_>>())
}

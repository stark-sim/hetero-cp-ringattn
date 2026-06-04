fn main() {
    let tokenizer_path = "/Users/stark_sim/VSCodeProjects/hetero-cp-ringattn/models/Qwen2-0.5B/tokenizer.json";
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
    let encoding = tokenizer.encode("Hello, world!", true).unwrap();
    println!("Rust tokens: {:?}", encoding.get_ids());
    for id in encoding.get_ids() {
        println!("  token {} -> '{}'", id, tokenizer.decode(&[*id], true).unwrap());
    }
}

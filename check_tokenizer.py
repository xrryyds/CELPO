import os
import sys
from transformers import AutoTokenizer

# ==========================================
# 在这里修改你的模型路径
MODEL_PATH = "/root/project/data/xrr/OREAL-7B" 
# ==========================================

def check_tokenizer(model_path):
    print(f"[*] Starting diagnosis for: {model_path}\n")

    # 1. 检查物理文件是否存在
    print("--- Phase 1: File Integrity Check ---")
    if not os.path.exists(model_path):
        print(f"[!] Error: Path does not exist: {model_path}")
        return

    files = os.listdir(model_path)
    # 常见的 tokenizer 文件
    expected_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    # 词表文件通常是 vocab.json + merges.txt 或者 tokenizer.model
    vocab_files = ["vocab.json", "merges.txt", "tokenizer.model"]
    
    found_any_vocab = False
    for f in expected_files:
        if f in files:
            print(f"[OK] Found: {f}")
        else:
            print(f"[Warning] Missing typical config file: {f}")

    for f in vocab_files:
        if f in files:
            print(f"[OK] Found vocab file: {f}")
            found_any_vocab = True
    
    if not found_any_vocab:
        print("[!] Critical Warning: No vocabulary file (vocab.json/tokenizer.model) found! Model might fail to load.")
    print("")

    # 2. 尝试加载 Tokenizer
    print("--- Phase 2: Loading Test ---")
    try:
        # trust_remote_code=True 是为了防止某些自定义模型（如 Qwen/ChatGLM）加载失败
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded successfully.")
        print(f"     Tokenizer Class: {tokenizer.__class__.__name__}")
        print(f"     Vocab Size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"[!] Fatal Error: Failed to load tokenizer.")
        print(f"    Error details: {e}")
        return

    # 3. 检查特殊 Token
    print("\n--- Phase 3: Special Tokens Inspection ---")
    special_tokens = {
        "PAD": (tokenizer.pad_token, tokenizer.pad_token_id),
        "EOS": (tokenizer.eos_token, tokenizer.eos_token_id),
        "BOS": (tokenizer.bos_token, tokenizer.bos_token_id),
        "UNK": (tokenizer.unk_token, tokenizer.unk_token_id),
    }

    for name, (token, token_id) in special_tokens.items():
        status = "OK" if token is not None else "None (Check if this is expected)"
        print(f"{name:<4}: Token='{token}' | ID={token_id} | Status: {status}")

    # 4. 编码/解码 回环测试 (Round-Trip Test)
    print("\n--- Phase 4: Round-Trip Functionality Test ---")
    test_text = "Hello, world! 1+1=2. 这是一个测试。"
    try:
        # Encode
        encoded_ids = tokenizer.encode(test_text, add_special_tokens=False)
        print(f"Input Text:  {test_text}")
        print(f"Encoded IDs: {encoded_ids}")
        
        # Decode
        decoded_text = tokenizer.decode(encoded_ids)
        print(f"Decoded Txt: {decoded_text}")

        # 验证一致性 (忽略空格差异，因为某些 tokenizer 会处理前导空格)
        if test_text.replace(" ", "") == decoded_text.replace(" ", ""):
            print("[OK] Decode matches Encode (ignoring spaces).")
        elif test_text in decoded_text:
            print("[OK] Input is contained in Output.")
        else:
            print("[!] Warning: Decoded text does not perfectly match input. This might be normal for some tokenizers but check carefully.")
            
    except Exception as e:
        print(f"[!] Error during encoding/decoding: {e}")

    print("\n[*] Diagnosis Complete.")

if __name__ == "__main__":
    check_tokenizer(MODEL_PATH)

import tkinter as tk
from tkinter import ttk
import threading

# conversation_inference.py
import torch
import json
import re
from tokenizers import Tokenizer
import math
import numpy

# ===================== Model Architecture =====================
class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.register_buffer("cos", None)
        self.register_buffer("sin", None)
        self.build_cache(max_len)
    
    def build_cache(self, seq_len):
        if self.cos is not None and seq_len <= self.cos.size(0):
            return
            
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
        
        sin = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)
        
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
    
    def forward(self, x):
        seq_len = x.size(-2)
        self.build_cache(seq_len)
        
        sin = self.sin[:seq_len]
        cos = self.cos[:seq_len]
        
        view_shape = (1,) * (x.dim() - 2) + (seq_len, self.dim // 2)
        sin = sin.view(view_shape)
        cos = cos.view(view_shape)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        x_rot = torch.cat(
            [x1 * cos - x2 * sin,
             x1 * sin + x2 * cos],
            dim=-1
        )
        
        return x_rot

class ConversationalAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, num_heads):
        super().__init__()
        self.d_k = d_k
        self.num_heads = num_heads
        self.head_dim = d_k // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary positional encoding"
        
        self.qw_proj = torch.nn.Linear(d_model, d_k)
        self.kw_proj = torch.nn.Linear(d_model, d_k)
        self.qs_proj = torch.nn.Linear(d_model, d_k)
        self.ks_proj = torch.nn.Linear(d_model, d_k)
        self.v_proj = torch.nn.Linear(d_model, d_k)
        self.out_proj = torch.nn.Linear(d_k, d_model)
        
        self.rotary_pe = RotaryPositionalEncoding(self.head_dim)
        self.norm = torch.nn.LayerNorm(self.head_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

    def apply_rotary(self, x):
        return self.rotary_pe(x)

    def forward(self, word_embed, speaker_embed, mask=None):
        B, L, _ = word_embed.size()
        
        qw = self.qw_proj(word_embed)
        kw = self.kw_proj(word_embed)
        qs = self.qs_proj(speaker_embed)
        ks = self.ks_proj(speaker_embed)
        v = self.v_proj(word_embed)
        
        qw = self.split_heads(qw)
        kw = self.split_heads(kw)
        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        v = self.split_heads(v)
        
        qw = self.apply_rotary(qw)
        kw = self.apply_rotary(kw)
        qs = self.apply_rotary(qs)
        ks = self.apply_rotary(ks)
        
        qw = self.norm(qw)
        kw = self.norm(kw)
        qs = self.norm(qs)
        ks = self.norm(ks)
        
        content_attn = qw @ kw.transpose(-2, -1)
        speaker_attn = qs @ ks.transpose(-2, -1)
        
        scores = (content_attn + speaker_attn) / math.sqrt(2*self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        max_val = scores.amax(dim=-1, keepdim=True)
        scores = scores - max_val
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = self.combine_heads(out)
        out = self.out_proj(out)
        
        return out, attn

class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model, d_k, num_heads, ff_dim):
        super().__init__()
        self.self_attn = ConversationalAttention(d_model, d_k, num_heads)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(ff_dim, d_model),
            torch.nn.Dropout(0.2)
        )
        
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)

    def forward(self, x, speaker_embed, mask):
        attn_out, attn_weights = self.self_attn(x, speaker_embed, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        
        return x, attn_weights

class ConversationalTransformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, d_k, num_heads, ff_dim, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.speaker_embedding = torch.nn.Embedding(2, d_model)
        self.layers = torch.nn.ModuleList([
            DecoderBlock(d_model, d_k, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size)
        self.head.weight = self.embedding.weight
        
    def forward(self, input_ids, speaker_ids, mask=None):
        word_embed = self.embedding(input_ids)
        speaker_embed = self.speaker_embedding(speaker_ids)
        
        attn_weights = []
        x = word_embed
        for layer in self.layers:
            x, attn = layer(x, speaker_embed, mask)
            attn_weights.append(attn)
        
        x = self.final_norm(x)
        logits = self.head(x)
        
        return logits, attn_weights

# ===================== Utility Functions =====================
def clean_text(text):
    """Clean text for tokenization"""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:?!\'"-]', '', text)
    return text.strip()

def create_decoder_mask(seq_len, device):
    """Create causal mask for decoder"""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

# ===================== Model Loading =====================
def load_conversational_model(model_dir="C:\College\Custom - Transformer\Last Trained ChatBot - Conversational Model", device=None):
    """
    Load the trained conversational model
    Returns: (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    
    # Create model
    model = ConversationalTransformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        d_k=config['d_k'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        vocab_size=config['vocab_size']
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(f"{model_dir}/model_weights.pth", map_location=device))
    model.eval()
    
    return model, tokenizer, config

# ===================== Generation Function =====================
def generate_response(model, tokenizer, context, speaker=0, max_length=100, 
                     temperature=0.7, top_k=50, top_p=0.9, device=None):
    """
    Generate conversational response to given context
    
    Args:
        model: Loaded conversation model
        tokenizer: Loaded tokenizer
        context: List of conversation utterances (strings)
        speaker: Starting speaker (0 or 1)
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to use (auto-detected if None)
        
    Returns:
        List of generated utterances (strings)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Token IDs for special tokens
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    spk1_id = tokenizer.token_to_id("[SPK1]")
    spk2_id = tokenizer.token_to_id("[SPK2]")
    
    tokens = [bos_id]
    speakers = [0]  # Default speaker for BOS
    
    # Add context with speaker tokens
    for utterance in context:
        # Add speaker token
        spk_token = spk1_id if speaker == 0 else spk2_id
        tokens.append(spk_token)
        speakers.append(speaker)
        
        # Tokenize utterance
        utterance_ids = tokenizer.encode(clean_text(utterance)).ids
        tokens.extend(utterance_ids)
        speakers.extend([speaker] * len(utterance_ids))
        
        # Add EOS and switch speaker
        tokens.append(eos_id)
        speakers.append(speaker)
        speaker = 1 - speaker
    
    generated = tokens.copy()
    current_speakers = speakers.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = torch.tensor([generated[-config['seq_len']:]]).to(device)
            speaker_tensor = torch.tensor([current_speakers[-config['seq_len']:]]).to(device)
            seq_len = input_tensor.size(1)
            
            # Create causal mask
            causal_mask = create_decoder_mask(seq_len, device).unsqueeze(0)
            
            logits, _ = model(input_tensor, speaker_tensor, causal_mask)
            
            # Get last token logits
            next_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = next_logits.topk(top_k)
                min_val = top_k_logits[-1]
                next_logits[next_logits < min_val] = float('-inf')
            
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token using PyTorch (no NumPy needed)
            probabilities = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            # Append token and speaker
            generated.append(next_token)
            current_speakers.append(speaker)  # Current speaker for the new token
            
            # Stop if EOS is generated
            if next_token == eos_id:
                break
    
    # Decode tokens
    decoded = tokenizer.decode([t for t in generated if t not in [bos_id, eos_id, pad_id]])
    
    # Split into utterances using speaker tokens
    utterances = []
    current_utterance = []
    for token in generated:
        if token in [spk1_id, spk2_id]:
            if current_utterance:
                utterances.append(tokenizer.decode(current_utterance))
                current_utterance = []
        elif token not in [bos_id, eos_id, pad_id]:
            current_utterance.append(token)
    
    if current_utterance:
        utterances.append(tokenizer.decode(current_utterance))
    
    # Return only the generated responses (after the initial context)
    return utterances[len(context):]

class ChatBotGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Conversational AI Chatbot - SHAMS")
        self.root.geometry("500x600")

        # Load model once
        self.model, self.tokenizer, self.config = load_conversational_model()
        self.context = []
        self.speaker = 0  # 0 = user, 1 = AI

        # ================= UI =================
        self.chat_area = tk.Text(
            self.root,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Arial", 11)
        )
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.pack(fill=tk.X, padx=10, pady=5)

        self.user_input = tk.Entry(self.entry_frame, font=("Arial", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            self.entry_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        self.display_message("System", "Chatbot loaded successfully!")

        self.root.mainloop()

    # ================= Helper Functions =================
    def display_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.user_input.delete(0, tk.END)
        self.display_message("You", user_text)

        # Add user input to context
        self.context.append(user_text)

        # Generate response in a separate thread (prevents UI freeze)
        threading.Thread(target=self.generate_reply).start()

    def generate_reply(self):
        responses = generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            context=self.context,
            speaker=self.speaker,
            max_length=100,
            temperature=0.8
        )

        for response in responses:
            self.display_message("AI", response)
            self.context.append(response)

        # Switch speaker
        self.speaker = 1 - self.speaker


if __name__ == "__main__":
    ChatBotGUI()

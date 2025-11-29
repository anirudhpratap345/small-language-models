"""
Diving Deep into Small Language Models (SLMs)
==============================================

This module explores the fundamental workings of small language models, including:
- Tokenization & Embeddings: Converting text to numerical representations
- Transformer Architecture: The backbone of modern language models
- Attention Mechanisms: How models focus on relevant information
- Text Generation: Autoregressive decoding and sampling strategies
- Evaluation Metrics: Assessing model performance

Author: AI Assistant
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


# =============================================================================
# SECTION 1: TOKENIZATION AND EMBEDDINGS
# =============================================================================

def demonstrate_tokenization():
    """
    Demonstrates how text is converted to tokens and embeddings.
    
    Tokenization: Breaking text into subword units
    Embeddings: Mapping tokens to dense numerical vectors
    """
    print("\n" + "=" * 70)
    print("SECTION 1: TOKENIZATION AND EMBEDDINGS")
    print("=" * 70)
    
    # Load a tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Sample text
    sample_text = "Small language models are efficient and powerful!"
    
    # Tokenize the text
    tokens = tokenizer.encode(sample_text)
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    print(f"\nOriginal text: {sample_text}")
    print(f"Tokens (IDs): {tokens}")
    print(f"Token strings: {token_strings}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of unique tokens: {len(set(tokens))}")
    
    vocab_size = tokenizer.vocab_size
    print(f"GPT-2 vocabulary size: {vocab_size}")
    
    # Visualize tokenization
    fig, ax = plt.subplots(figsize=(12, 4))
    
    for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
        ax.barh(i, 1, left=0, height=0.8, color=plt.cm.viridis(token_id/vocab_size))
        ax.text(0.5, i, f"{token_str} ({token_id})", va='center', ha='center',
                color='white', fontweight='bold', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(tokens)-0.5)
    ax.set_ylabel('Token Position', fontsize=11)
    ax.set_title('Tokenization Process: Text to Token IDs', fontsize=13, fontweight='bold')
    ax.set_xticks([])
    plt.tight_layout()
    plt.show()
    
    print("✓ Each token is mapped to a unique integer ID in the vocabulary")
    
    # Embedding Layer Demonstration
    embedding_dim = 8  # Small dimension for visualization
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    
    token_tensor = torch.tensor(tokens)
    embeddings = embedding_layer(token_tensor)
    
    print(f"\n--- Embeddings ---")
    print(f"Token IDs shape: {token_tensor.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"\nFirst token '{token_strings[0]}' embedding vector:")
    print(embeddings[0].detach().numpy())
    
    # Visualize embeddings
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(embeddings.detach().numpy(), aspect='auto', cmap='coolwarm')
    ax.set_xlabel('Embedding Dimension', fontsize=11)
    ax.set_ylabel('Token Position', fontsize=11)
    ax.set_title(f'Token Embeddings Heatmap (dim={embedding_dim})', fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(token_strings)))
    ax.set_yticklabels(token_strings, fontsize=9)
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    plt.show()
    
    print("✓ Each token is now represented as a dense vector in embedding space")
    
    return tokenizer, embeddings


# =============================================================================
# SECTION 2: TRANSFORMER ARCHITECTURE
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding adds position information to embeddings.
    
    Uses sine and cosine functions with different frequencies to encode positions.
    This allows the model to understand the relative positions of tokens.
    """
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Sine and cosine functions with different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


def demonstrate_positional_encoding():
    """Demonstrates and visualizes positional encoding."""
    print("\n" + "=" * 70)
    print("SECTION 2: TRANSFORMER ARCHITECTURE - POSITIONAL ENCODING")
    print("=" * 70)
    
    d_model = 32
    pos_encoder = PositionalEncoding(d_model, max_len=20)
    pos_encoding = pos_encoder.pe[:10, :]
    
    print(f"Positional encoding dimension: {d_model}")
    print(f"Max sequence length: 20")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Heatmap of positional encoding
    im = axes[0].imshow(pos_encoding.T, aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Position', fontsize=11)
    axes[0].set_ylabel('Dimension', fontsize=11)
    axes[0].set_title('Positional Encoding Pattern', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0])
    
    # Pattern visualization
    for i in range(4):
        axes[1].plot(pos_encoding[:, i].numpy(), label=f'Dim {i}', marker='o')
    axes[1].set_xlabel('Position', fontsize=11)
    axes[1].set_ylabel('Encoding Value', fontsize=11)
    axes[1].set_title('Positional Encoding Values', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Positional encoding provides position information to the model")


# =============================================================================
# SECTION 3: ATTENTION MECHANISM
# =============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism implementation.
    
    Allows each token to attend to all other tokens in the sequence.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Where:
    - Q (Query): What the token is looking for
    - K (Key): What information each token has
    - V (Value): The actual information to pass
    """
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Attention output
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back and apply final linear layer
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.fc_out(context)
        
        return output, attention_weights


def demonstrate_attention():
    """Demonstrates and visualizes the attention mechanism."""
    print("\n" + "=" * 70)
    print("SECTION 3: ATTENTION MECHANISM")
    print("=" * 70)
    
    batch_size = 1
    seq_len = 5
    d_model = 16
    
    sample_input = torch.randn(batch_size, seq_len, d_model)
    attention = SelfAttention(d_model, num_heads=2)
    
    output, attn_weights = attention(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Number of attention heads: 2")
    
    # Visualize attention weights
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for head_idx in range(2):
        attn_map = attn_weights[0, head_idx].detach().numpy()
        im = axes[head_idx].imshow(attn_map, cmap='YlOrRd', vmin=0, vmax=1)
        axes[head_idx].set_xlabel('Key Position', fontsize=10)
        axes[head_idx].set_ylabel('Query Position', fontsize=10)
        axes[head_idx].set_title(f'Attention Weights - Head {head_idx + 1}', fontweight='bold')
        plt.colorbar(im, ax=axes[head_idx])
        
        # Add text annotations
        for i in range(seq_len):
            for j in range(seq_len):
                axes[head_idx].text(j, i, f'{attn_map[i, j]:.2f}',
                                  ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Attention weights show how each position attends to all other positions")


# =============================================================================
# SECTION 4: FORWARD PASS THROUGH A SMALL LANGUAGE MODEL
# =============================================================================

def demonstrate_forward_pass():
    """Traces data flow through a pre-trained small language model."""
    print("\n" + "=" * 70)
    print("SECTION 4: FORWARD PASS THROUGH A SMALL LANGUAGE MODEL")
    print("=" * 70)
    
    # Load a small pre-trained model
    print("\nLoading DistilGPT-2 model...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    
    # Prepare input
    text = "Machine learning is"
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Input text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    
    logits = outputs.logits
    hidden_states = outputs.hidden_states
    
    print(f"\nModel outputs:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Number of hidden layers: {len(hidden_states)}")
    print(f"  Hidden state shape: {hidden_states[0].shape}")
    
    # Show how logits transform to probabilities
    next_token_logits = logits[0, -1, :]
    next_token_probs = F.softmax(next_token_logits, dim=-1)
    
    # Get top-5 predictions
    top_k = 5
    top_probs, top_indices = torch.topk(next_token_probs, top_k)
    
    print(f"\nTop-{top_k} next token predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        token_str = tokenizer.decode([idx])
        print(f"  {i}. {token_str:20s} - Probability: {prob.item():.4f}")
    
    # Visualize the flow through layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Logits distribution
    axes[0, 0].hist(logits[0, -1, :].detach().cpu().numpy(), bins=50,
                    color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Logit Value', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Logits for Next Token', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Probability distribution (top 20)
    top_20_probs, top_20_indices = torch.topk(next_token_probs, 20)
    token_strings = [tokenizer.decode([idx]).strip() or f"[{idx}]" for idx in top_20_indices]
    axes[0, 1].barh(range(20), top_20_probs.detach().cpu().numpy(), color='coral')
    axes[0, 1].set_yticks(range(20))
    axes[0, 1].set_yticklabels(token_strings, fontsize=9)
    axes[0, 1].set_xlabel('Probability', fontsize=10)
    axes[0, 1].set_title('Top-20 Next Token Predictions', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # 3. Hidden state magnitudes across layers
    layer_magnitudes = []
    for hidden in hidden_states:
        magnitude = torch.norm(hidden[0, -1, :]).item()
        layer_magnitudes.append(magnitude)
    
    axes[1, 0].plot(layer_magnitudes, marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Layer', fontsize=10)
    axes[1, 0].set_ylabel('Hidden State Magnitude', fontsize=10)
    axes[1, 0].set_title('Hidden State Magnitudes Across Layers', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Token embedding progression
    token_embeddings = []
    for hidden in hidden_states:
        embedding = hidden[0, -1, :].detach().cpu().numpy()
        token_embeddings.append(embedding[:32])  # First 32 dimensions
    
    token_embeddings = np.array(token_embeddings)
    im = axes[1, 1].imshow(token_embeddings, aspect='auto', cmap='coolwarm')
    axes[1, 1].set_xlabel('Embedding Dimension', fontsize=10)
    axes[1, 1].set_ylabel('Layer', fontsize=10)
    axes[1, 1].set_title('Token Embedding Evolution Across Layers', fontweight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='Value')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Data flows through embedding → attention layers → MLP → output projections")
    
    return tokenizer, model


# =============================================================================
# SECTION 5: TEMPERATURE AND SAMPLING
# =============================================================================

def top_k_sampling(logits, k=5, temperature=1.0):
    """Select from top-k most likely tokens."""
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    return probs, top_k_indices


def top_p_sampling(logits, p=0.9, temperature=1.0):
    """Select from smallest set of tokens with cumulative probability >= p."""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Remove tokens above cumulative probability threshold
    sorted_indices_to_remove = cumsum_probs > p
    sorted_indices_to_remove[0] = False  # Keep at least one token
    
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    return sorted_probs, sorted_indices


def demonstrate_temperature_and_sampling(logits, tokenizer):
    """Demonstrates temperature effects and different sampling strategies."""
    print("\n" + "=" * 70)
    print("SECTION 5: TEMPERATURE AND SAMPLING")
    print("=" * 70)
    
    # Temperature effects
    temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]
    original_probs = F.softmax(logits[0, -1, :], dim=-1).detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    
    for idx, temp in enumerate(temperatures):
        scaled_logits = logits[0, -1, :] / temp
        temp_probs = F.softmax(scaled_logits, dim=-1).detach().cpu().numpy()
        
        top_indices = np.argsort(-temp_probs)[:10]
        top_probs_vals = temp_probs[top_indices]
        token_labels = [tokenizer.decode([i]).strip() or f"[{i}]" for i in top_indices]
        
        axes[idx].barh(range(10), top_probs_vals,
                      color=plt.cm.viridis(idx/len(temperatures)))
        axes[idx].set_yticks(range(10))
        axes[idx].set_yticklabels(token_labels, fontsize=8)
        axes[idx].set_xlim(0, max(top_probs_vals) * 1.1)
        axes[idx].set_title(f'T={temp}', fontweight='bold')
        axes[idx].set_xlabel('Probability', fontsize=9)
        axes[idx].invert_yaxis()
        if idx == 0:
            axes[idx].set_ylabel('Tokens', fontsize=9)
    
    plt.tight_layout()
    plt.suptitle('Effect of Temperature on Probability Distribution',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.show()
    
    print("\nTemperature Effects:")
    print("  Low Temperature (0.3):   Sharp distribution, deterministic")
    print("  High Temperature (2.0):  Flat distribution, more random")
    
    # Sampling strategies comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    all_probs = F.softmax(logits[0, -1, :], dim=-1).detach().cpu().numpy()
    top_indices_all = np.argsort(-all_probs)[:15]
    top_probs_all = all_probs[top_indices_all]
    token_labels = [tokenizer.decode([i]).strip() or f"[{i}]" for i in top_indices_all]
    
    axes[0].barh(range(15), top_probs_all, color='skyblue')
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(token_labels, fontsize=9)
    axes[0].set_title('Greedy Decoding\n(All tokens considered)', fontweight='bold')
    axes[0].set_xlabel('Probability', fontsize=10)
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis='x')
    
    # Top-K sampling
    top_k_probs, top_k_idx = top_k_sampling(logits[0, -1, :], k=5)
    top_k_probs = top_k_probs.detach().cpu().numpy()
    top_k_idx = top_k_idx.detach().cpu().numpy()
    token_labels_k = [tokenizer.decode([i]).strip() or f"[{i}]" for i in top_k_idx]
    
    axes[1].barh(range(5), top_k_probs, color='lightcoral')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(token_labels_k, fontsize=9)
    axes[1].set_title('Top-K Sampling (K=5)\n(Only 5 best tokens)', fontweight='bold')
    axes[1].set_xlabel('Probability', fontsize=10)
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3, axis='x')
    
    # Top-P sampling
    top_p_probs, top_p_idx = top_p_sampling(logits[0, -1, :], p=0.9)
    top_p_probs = top_p_probs.detach().cpu().numpy()
    top_p_idx_nonzero = top_p_idx[top_p_probs > 0]
    top_p_probs_nonzero = top_p_probs[top_p_probs > 0]
    token_labels_p = [tokenizer.decode([i]).strip() or f"[{i}]" for i in top_p_idx_nonzero]
    
    axes[2].barh(range(len(token_labels_p)), top_p_probs_nonzero, color='lightgreen')
    axes[2].set_yticks(range(len(token_labels_p)))
    axes[2].set_yticklabels(token_labels_p, fontsize=9)
    axes[2].set_title(f'Nucleus Sampling (P=0.9)\n({len(token_labels_p)} tokens)', fontweight='bold')
    axes[2].set_xlabel('Probability', fontsize=10)
    axes[2].invert_yaxis()
    axes[2].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Different sampling strategies balance quality and diversity")


# =============================================================================
# SECTION 6: TEXT GENERATION
# =============================================================================

def demonstrate_text_generation(tokenizer, model):
    """Generates text using different strategies and parameters."""
    print("\n" + "=" * 70)
    print("SECTION 6: TEXT GENERATION")
    print("=" * 70)
    
    prompts = [
        "Artificial intelligence is",
        "The future of technology will",
        "Deep learning models are"
    ]
    
    generation_configs = {
        "Greedy": {
            "do_sample": False,
            "max_new_tokens": 20,
            "temperature": 1.0
        },
        "Temperature=0.5": {
            "do_sample": True,
            "max_new_tokens": 20,
            "temperature": 0.5,
            "top_p": 0.95
        },
        "Temperature=1.5": {
            "do_sample": True,
            "max_new_tokens": 20,
            "temperature": 1.5,
            "top_p": 0.95
        },
        "Top-K=5": {
            "do_sample": True,
            "max_new_tokens": 20,
            "temperature": 1.0,
            "top_k": 5
        }
    }
    
    print("\nTEXT GENERATION EXAMPLES")
    print("-" * 70)
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 70)
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        for config_name, config in generation_configs.items():
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    **config,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_continuation = generated_text[len(prompt):]
            
            print(f"  [{config_name:15s}]: {generated_continuation}")


# =============================================================================
# SECTION 7: STEP-BY-STEP GENERATION ANALYSIS
# =============================================================================

def demonstrate_step_by_step_generation(tokenizer, model):
    """Visualizes the autoregressive generation process step-by-step."""
    print("\n" + "=" * 70)
    print("SECTION 7: STEP-BY-STEP GENERATION ANALYSIS")
    print("=" * 70)
    
    prompt = "Machine learning is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"Prompt: {prompt}")
    print("-" * 70)
    
    generated_sequence = input_ids.clone()
    token_sequence = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    probs_sequence = []
    entropy_sequence = []
    
    for step in range(5):  # Generate 5 tokens
        with torch.no_grad():
            outputs = model(generated_sequence)
        
        next_token_logits = outputs.logits[0, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        # Calculate entropy
        entropy = -(next_token_probs * torch.log(next_token_probs + 1e-10)).sum()
        entropy_sequence.append(entropy.item())
        
        # Sample next token
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)], dim=1)
        
        token_str = tokenizer.decode(next_token.item())
        token_sequence.append(token_str)
        
        top_prob = next_token_probs.max().item()
        probs_sequence.append(top_prob)
        
        full_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        print(f"Step {step+1}: Added '{token_str:10s}' | Max prob: {top_prob:.4f} | Text: {full_text}")
    
    # Visualize the process
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = list(range(1, 6))
    
    # Max probability over steps
    axes[0].plot(steps, probs_sequence, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0].set_xlabel('Generation Step', fontsize=11)
    axes[0].set_ylabel('Max Token Probability', fontsize=11)
    axes[0].set_title('Model Confidence During Generation', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    
    # Entropy over steps
    axes[1].plot(steps, entropy_sequence, marker='s', linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Generation Step', fontsize=11)
    axes[1].set_ylabel('Entropy of Distribution', fontsize=11)
    axes[1].set_title('Distribution Uncertainty During Generation', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Generation is an autoregressive process: each new token depends on all previous tokens")


# =============================================================================
# SECTION 8: MODEL EVALUATION METRICS
# =============================================================================

def demonstrate_evaluation_metrics(tokenizer, model):
    """Calculates and interprets perplexity and log-likelihood metrics."""
    print("\n" + "=" * 70)
    print("SECTION 8: MODEL EVALUATION METRICS")
    print("=" * 70)
    
    evaluation_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Python is a popular programming language for data science."
    ]
    
    results = []
    
    for text in evaluation_texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        
        loss = outputs.loss.item()
        perplexity = np.exp(loss)
        
        results.append({
            'text': text,
            'loss': loss,
            'perplexity': perplexity,
            'length': input_ids.shape[1]
        })
        
        print(f"\nText: {text[:50]}...")
        print(f"  Loss:       {loss:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Seq Length: {input_ids.shape[1]}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    texts_short = [r['text'][:40] + "..." for r in results]
    losses = [r['loss'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    
    # Loss
    axes[0].bar(range(len(texts_short)), losses, color='lightblue', edgecolor='black')
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Language Modeling Loss', fontweight='bold')
    axes[0].set_xticks(range(len(texts_short)))
    axes[0].set_xticklabels(texts_short, rotation=45, ha='right', fontsize=9)
    axes[0].grid(alpha=0.3, axis='y')
    
    # Perplexity
    axes[1].bar(range(len(texts_short)), perplexities, color='lightcoral', edgecolor='black')
    axes[1].set_ylabel('Perplexity', fontsize=11)
    axes[1].set_title('Perplexity Score', fontweight='bold')
    axes[1].set_xticks(range(len(texts_short)))
    axes[1].set_xticklabels(texts_short, rotation=45, ha='right', fontsize=9)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Lower perplexity indicates the model better predicts the text")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DIVING DEEP INTO SMALL LANGUAGE MODELS")
    print("=" * 70)
    print("This script explores the fundamental workings of SLMs through")
    print("interactive demonstrations and visualizations.")
    print("=" * 70)
    
    # Section 1: Tokenization and Embeddings
    tokenizer_demo, embeddings = demonstrate_tokenization()
    
    # Section 2: Positional Encoding
    demonstrate_positional_encoding()
    
    # Section 3: Attention Mechanism
    demonstrate_attention()
    
    # Section 4: Forward Pass
    tokenizer, model = demonstrate_forward_pass()
    
    # Get logits for sampling demonstrations
    text = "Machine learning is"
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    
    # Section 5: Temperature and Sampling
    demonstrate_temperature_and_sampling(logits, tokenizer)
    
    # Section 6: Text Generation
    demonstrate_text_generation(tokenizer, model)
    
    # Section 7: Step-by-step Generation
    demonstrate_step_by_step_generation(tokenizer, model)
    
    # Section 8: Evaluation Metrics
    results = demonstrate_evaluation_metrics(tokenizer, model)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways about Small Language Models:

1. TOKENIZATION: Text is converted to subword tokens, then to embeddings
2. ARCHITECTURE: Transformers use multi-head attention and feed-forward networks
3. ATTENTION: Each token attends to all others, learning contextual relationships
4. FORWARD PASS: Data flows through layers, with representations refined at each step
5. TEMPERATURE: Controls randomness in predictions - higher = more creative
6. SAMPLING: Different strategies (greedy, top-k, nucleus) balance quality and diversity
7. GENERATION: Autoregressive decoding predicts one token at a time
8. EVALUATION: Perplexity and loss measure how well the model predicts text

These mechanisms work together to enable language understanding and generation!
""")
    print("=" * 70)


if __name__ == "__main__":
    main()

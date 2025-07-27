import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
# Define the list of models to be tested. Each model is a dictionary with a 'name' key.
MODELS_TO_TEST = [
    {"name": "gpt2"},
    {"name": "Qwen/Qwen2-0.5B-Instruct"},
    {"name": "mistralai/Mistral-7B-Instruct-v0.2"},
    {"name": "microsoft/Phi-3-mini-4k-instruct"},
    # {"name": "microsoft/Phi-4-mini-reasoning"}, # Example of a larger model you might add
    {"name": "NousResearch/Hermes-3-Llama-3.1-8B"},
    {"name": "Qwen/Qwen3-8B"},
    {"name": "Qwen/Qwen3-4B"},
]

# A list of arbitrary, complex strings used as prompts to probe the model's internal state.
# These are designed to be content-neutral and difficult to compress.
PROBE_PROMPTS = [
    "80170451589413662094171256883326",
    "13114447058729740763706819138784",
    "23679353403277139303706311123210",
    "92549256953980330378091746680898",
    "35800111737760281045089078690774",
]

# Set the computation device. Use CUDA if available, otherwise CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- CALCULATION FUNCTIONS ---

def get_gradient_pairs(model, tokenizer, layers, prompt):
    """
    Performs a single forward and backward pass to collect gradient pairs and loss.

    This function uses backward hooks on specified layers to intercept the gradients
    as they flow through the network. It captures the gradient with respect to a
    layer's input (ΔA) and the gradient with respect to its output (ΔB).

    Args:
        model (AutoModelForCausalLM): The language model to be analyzed.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        layers (list): A list of the transformer layer modules to hook into.
        prompt (str): The input text for the forward/backward pass.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dict holds a {'ΔA', 'ΔB'} pair
                    for a layer. The list is ordered from the first to the last layer.
            - float: The loss value for the pass.
    """
    gradient_pairs = []

    def collecting_hook(module, grad_input, grad_output):
        """A hook function to capture gradients."""
        # Ensure both input and output gradients exist before processing.
        if grad_input and grad_input[0] is not None and grad_output and grad_output[0] is not None:
            # Detach and clone the tensors to prevent holding onto the computation graph
            # and to ensure they are not modified by subsequent operations.
            gradient_pairs.append({
                "ΔA": grad_input[0].detach().clone(),
                "ΔB": grad_output[0].detach().clone()
            })

    # Register the hook on all specified layers.
    hook_handles = [layer.register_full_backward_hook(collecting_hook) for layer in layers]
    loss_value = None

    try:
        model.zero_grad()  # Clear any existing gradients.
        # Tokenize the prompt and prepare for the model.
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(model.device)
        labels = inputs.input_ids.clone()
        # In language modeling, we ignore padding tokens in the loss calculation.
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss_value = loss.item()
        
        # Backward pass to compute gradients
        loss.backward()
    finally:
        # CRITICAL: Always remove hooks to prevent memory leaks and unintended behavior.
        for handle in hook_handles:
            handle.remove()
            
    # Hooks are executed in reverse order of layers. Reverse the list to match forward-pass order.
    return list(reversed(gradient_pairs)), loss_value

def calculate_all_metrics(model, tokenizer, layers, prompts):
    """
    Calculates all 5 metrics for the Internal Physics Benchmark (IPB).

    This function orchestrates the gradient collection for multiple prompts and computes
    the final average scores for EC, ISC, ISG, ISA, and ISF.

    Args:
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The model's tokenizer.
        layers (list): The list of transformer layers to analyze.
        prompts (list): A list of strings to use as input.

    Returns:
        dict: A dictionary containing the final scores for each of the 5 metrics.
              Returns NaNs if gradient collection fails.
    """
    print("  Calculating the full Internal Physics Benchmark (IPB)...")

    all_grad_pairs_per_prompt = []
    all_losses = []

    for prompt in prompts:
        grad_pairs, loss_val = get_gradient_pairs(model, tokenizer, layers, prompt)
        if grad_pairs:
            all_grad_pairs_per_prompt.append(grad_pairs)
            all_losses.append(loss_val)

    if not all_grad_pairs_per_prompt:
        print("    Warning: Failed to collect any gradient pairs.")
        return {"ec_score": np.nan, "isc_score": np.nan, "isg_score": np.nan, "isa_score": np.nan, "isf_score": np.nan}

    # Lists to collect the average scores for each prompt
    ec_scores, isc_scores, isg_scores, isa_scores = [], [], [], []

    for i, grad_pairs in enumerate(all_grad_pairs_per_prompt):
        loss = all_losses[i]

        # 1. EC (Error of Conservation) Score
        # For each layer, calculate the normalized difference between input and output gradients.
        # .item() is crucial to move the result to CPU as a float, preventing GPU memory accumulation.
        ec_per_layer = [(torch.norm(p['ΔA'] - p['ΔB']) / (torch.norm(p['ΔB']) + 1e-9)).item() for p in grad_pairs]
        ec_scores.append(np.mean(ec_per_layer))

        # 2. ISC (Index of Computational Strain) Score
        # Total gradient "energy" (Frobenius norm of input gradients) divided by the loss.
        total_energy = sum(torch.norm(p['ΔA'], p='fro').item() for p in grad_pairs)
        isc_scores.append(total_energy / loss if loss > 1e-9 else 0)

        # 3. ISG (Index of Magnitude Synchronization) Score
        # Coefficient of variation (std/mean) of the EC scores across layers.
        mean_ec, std_ec = np.mean(ec_per_layer), np.std(ec_per_layer)
        isg_scores.append(std_ec / mean_ec if mean_ec > 1e-9 else 0)

        # 4. ISA (Index of Angular Synchronization) Score
        # Standard deviation of the cosine similarity between input and output gradients across layers.
        cos_sims = [cosine_similarity(p['ΔA'].flatten(), p['ΔB'].flatten(), dim=0).item() for p in grad_pairs]
        isa_scores.append(np.std(cos_sims))

    # 5. ISF (Index of Field Structure) Score
    try:
        # Create a matrix where each row is the flattened global gradient state for one prompt.
        X_data_for_isf = np.array([
            torch.cat([p['ΔA'].flatten().cpu() for p in grad_pairs]).numpy() for grad_pairs in all_grad_pairs_per_prompt
        ])
        
        # Use PCA to find the principal components of the global gradient state space.
        # This reveals the dominant correlation structures.
        pca = PCA(n_components=min(X_data_for_isf.shape))
        pca.fit(X_data_for_isf - np.mean(X_data_for_isf, axis=0)) # Center the data
        eigenvalues = pca.explained_variance_

        # Fit the eigenvalue spectrum to a power law: y = a * x^(-k)
        def power_law(x, a, k): return a * np.power(x, -k)
        
        y_data = sorted(eigenvalues, reverse=True)
        x_data = np.arange(1, len(y_data) + 1)
        
        params, _ = curve_fit(power_law, x_data, y_data, p0=[y_data[0], 1.0], maxfev=5000)
        y_fit = power_law(x_data, *params)
        
        # The score is the normalized Root Mean Squared Error of the fit.
        # A low score means the data fits a power law well.
        rmse = np.sqrt(np.mean((y_data - y_fit) ** 2))
        data_range = np.max(y_data) - np.min(y_data)
        isf_score = rmse / (data_range + 1e-9) if data_range > 1e-9 else 0.0

    except Exception as e:
        print(f"    Could not calculate ISF score: {e}")
        isf_score = np.nan

    # Return the final metrics, averaged across all prompts.
    return {
        "ec_score": np.mean(ec_scores),
        "isc_score": np.mean(isc_scores),
        "isg_score": np.mean(isg_scores),
        "isa_score": np.mean(isa_scores),
        "isf_score": isf_score
    }

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    all_model_results = []
    # Configure 4-bit quantization to load large models on consumer GPUs.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    results_file = "ipb_results.csv"

    # Robustness: Check for existing results to avoid re-running tested models.
    try:
        existing_df = pd.read_csv(results_file)
        existing_models = existing_df['model_name'].tolist()
    except FileNotFoundError:
        existing_models = []

    for config in MODELS_TO_TEST:
        model_name_short = config["name"].split('/')[-1]
        if model_name_short in existing_models:
            print(f"\n--- Skipping already tested model: {config['name']} ---")
            continue

        print(f"\n--- Processing model: {config['name']} ---")
        model, tokenizer = None, None # Ensure they are cleared in the finally block
        try:
            tokenizer = AutoTokenizer.from_pretrained(config["name"], trust_remote_code=True)
            # Some models don't have a pad token, so we use the EOS token instead.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load gpt2 without quantization, as it's small.
            if config["name"] == "gpt2":
                model = AutoModelForCausalLM.from_pretrained(
                    config["name"],
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32
                )
            else:
                # Load all other models with 4-bit quantization.
                model = AutoModelForCausalLM.from_pretrained(
                    config["name"],
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=quantization_config
                )
            
            model.eval() # Set model to evaluation mode.

            # This part robustly finds the list of transformer layers.
            # Handles different model architectures (e.g., Llama vs. GPT-2).
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers # For Llama, Mistral, Qwen, etc.
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h # For GPT-2
            else:
                raise TypeError(f"Could not determine the transformer layers for {config['name']}")

            print(f"Found {len(layers)} transformer layers.")

            metrics = calculate_all_metrics(model, tokenizer, layers, PROBE_PROMPTS)
            
            all_model_results.append({"model_name": model_name_short, **metrics})

        except Exception as e:
            print(f"FATAL ERROR processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Aggressively clean up memory before loading the next model.
            del model, tokenizer, layers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save results incrementally to avoid losing progress.
            if all_model_results:
                new_results_df = pd.DataFrame(all_model_results)
                try:
                    df_existing = pd.read_csv(results_file)
                    # Concatenate and remove duplicates, keeping the latest run.
                    pd.concat([df_existing, new_results_df]).drop_duplicates(
                        subset=['model_name'], keep='last'
                    ).to_csv(results_file, index=False)
                except FileNotFoundError:
                    new_results_df.to_csv(results_file, index=False)
                all_model_results = [] # Clear the list after saving

    # --- FINAL ANALYSIS AND VISUALIZATION ---
    print("\n--- Analysis of All Models ---")
    try:
        df = pd.read_csv(results_file).dropna()
        if df.empty:
            print("No results to analyze.")
        else:
            print("\n\n--- Internal Physics Benchmark Results ---")
            print(df.to_string())

            # --- Metric Explanations ---
            print("\n--- Metric Explanations (The Model's 'Personality') ---")
            print("Each metric describes a 'dimension' of the model's internal behavior:\n")
            print("EC (Error of Conservation):")
            print("  - Measures: The average magnitude of signal transformation (||ΔA - ΔB||).")
            print("  - Interpretation: How 'actively' the model modifies information. Low = transparent, High = transformative.\n")
            print("ISC (Index of Computational Strain):")
            print("  - Measures: The total gradient energy spent per unit of loss.")
            print("  - Interpretation: The model's energy efficiency. Low = efficient, High = wasteful.\n")
            print("ISG (Index of Magnitude Synchronization):")
            print("  - Measures: The variability of transformation magnitude across layers.")
            print("  - Interpretation: The operational style. Low = 'Roman Phalanx' (homogeneous), High = 'Swarm of Bees' (specialized).\n")
            print("ISA (Index of Angular Synchronization):")
            print("  - Measures: The variability of transformation direction across layers.")
            print("  - Interpretation: Directional coherence. Low = 'Crystal' (rigid), High = 'Ecosystem' (flexible).\n")
            print("ISF (Index of Field Structure):")
            print("  - Measures: How well the global correlation structure fits a power law.")
            print("  - Interpretation: Self-organization. Low = 'Gravitational Field' (long-range structure), High = 'Random Gas' (local correlations).\n")

            # --- Visualization ---
            if len(df) > 1:
                labels = ['Efficiency (1/ISC)', 'Conservation (1/EC)', 'Magnitude Sync (1/ISG)', 'Angular Sync (1/ISA)', 'Field Structure (1/ISF)']
                num_vars = len(labels)

                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1] # Close the circle

                fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
                
                # Normalize data for the plot. "Good" scores should be high.
                df_norm = df.copy()
                
                # Invert metrics where "lower is better" so that a higher value is always better for the plot.
                for col in ['isc_score', 'ec_score', 'isg_score', 'isa_score', 'isf_score']:
                    # Add a small epsilon to avoid division by zero.
                    df_norm[col] = 1 / (df[col] + 1e-9)
                
                # Scale each metric to a 0-1 range for fair comparison on the radar chart.
                for col in df_norm.columns[1:]: # Skip model_name
                    min_val, max_val = df_norm[col].min(), df_norm[col].max()
                    if (max_val - min_val) > 1e-9:
                        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                    else:
                        df_norm[col] = 0.5 # Assign a neutral value if all values for a metric are the same.

                for i, row in df_norm.iterrows():
                    stats = row.iloc[1:].values.tolist()
                    stats += stats[:1] # Close the loop
                    ax.plot(angles, stats, label=row.model_name, linewidth=2)
                    ax.fill(angles, stats, alpha=0.15)

                ax.set_yticklabels([])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, size=12)
                plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
                plt.title("Physical Personality Map of LLMs", size=16, y=1.1)
                plt.savefig("ipb_final_radar_chart.png", bbox_inches='tight')
                plt.show()

    except FileNotFoundError:
        print("Results file 'ipb_results.csv' not found. Please run the script to generate results.")

"""Text Generation by Thermodynamic Annealing

This is an experimental, novel approach to text generation.
Instead of autoregressive next-token prediction, we:

1. Initialize with a noisy/random distribution of token particles
2. Let thermodynamic dynamics pull them toward attractors
3. The attractors are shaped by the prompt/context
4. Read out the final token sequence

This is NOT diffusion in the DDPM sense. It's thermodynamic annealing.
The text "crystallizes" from noise via energy minimization.

Goal: Generate coherent text from prompt via thermodynamic annealing.
Metrics: Perplexity of generated text, coherence (measured by continuation)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.config import PhysicsConfig
from thermo_manifold.semantic.hierarchical import HierarchicalSemanticManifold

from .base import BaseExperiment, Scale


class TextDiffusionExperiment(BaseExperiment):
    """Text generation via thermodynamic annealing.
    
    EXPERIMENTAL: This is a novel approach, expect failures!
    """
    
    name = "text_diffusion"
    goal = "Generate text via thermodynamic annealing (experimental)"
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        super().__init__(scale, device, seed)
        
        # Scale-specific configs
        if scale == Scale.TOY:
            self.vocab_size = 500
            self.context_length = 16
            self.gen_length = 8
            self.annealing_steps = 50
        elif scale == Scale.MEDIUM:
            self.vocab_size = 5000
            self.context_length = 32
            self.gen_length = 16
            self.annealing_steps = 100
        else:
            self.vocab_size = 20000
            self.context_length = 64
            self.gen_length = 32
            self.annealing_steps = 200
        
        self.vocab: List[str] = []
        self.token_to_id: Dict[str, int] = {}
    
    def setup(self) -> None:
        """Load dataset and build vocabulary."""
        try:
            from datasets import load_dataset
            from collections import Counter
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"    Loading wikitext-2-raw-v1...")
        
        # `trust_remote_code` is no longer supported by `datasets` for security reasons.
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", streaming=True)
        
        # Build vocabulary
        print(f"    Building vocabulary...")
        word_counts: Counter = Counter()
        sample_count = 0
        
        for sample in dataset["train"]:
            text = sample["text"]
            if not text.strip():
                continue
            tokens = text.split()
            word_counts.update(tokens)
            sample_count += 1
            if sample_count >= 5000:
                break
        
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
        self.vocab = special_tokens + [
            word for word, _ in word_counts.most_common(self.vocab_size - len(special_tokens))
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        
        print(f"    Vocabulary size: {len(self.vocab)}")
        
        # Prefetch and tokenize data
        self._train_sequences: List[List[int]] = []
        self._eval_sequences: List[List[int]] = []
        
        self._tokenize_stream(dataset["train"], self._train_sequences, 2000)
        self._tokenize_stream(dataset["validation"], self._eval_sequences, 500)
        
        print(f"    Train sequences: {len(self._train_sequences)}")
        print(f"    Eval sequences: {len(self._eval_sequences)}")
        
        # Initialize manifold
        self.manifold = HierarchicalSemanticManifold(
            self.physics_config,
            self.device,
            vocab=self.vocab,
            embed_dim=min(self.scale_config.embed_dim, len(self.vocab)),
            chunk_min_len=2,
            chunk_max_len=4,
        )

    def _think_until_complete(self, *, max_steps: int = 20) -> None:
        """Run grammar steps until confidence-based halting or max_steps.

        Note: `thinking_complete()` can be slow to converge in general, so we cap
        the number of steps for experiments.
        """
        max_steps = int(max_steps)
        if max_steps <= 0:
            return
        for _ in range(max_steps):
            self.manifold.step_grammar()
            try:
                if self.manifold.thinking_complete():
                    break
            except Exception:
                # If a subclass doesn't support the halting test, fall back to fixed steps.
                break
    
    def _tokenize_stream(
        self,
        stream,
        output: List[List[int]],
        max_samples: int,
    ) -> None:
        """Tokenize stream into sequences."""
        unk_id = self.token_to_id.get("<unk>", 1)
        
        for sample in stream:
            text = sample["text"]
            if not text.strip():
                continue
            
            tokens = text.split()
            if len(tokens) < self.context_length + self.gen_length:
                continue
            
            ids = [self.token_to_id.get(t, unk_id) for t in tokens]
            output.append(ids)
            
            if len(output) >= max_samples:
                break
    
    def train_iterator(self) -> Iterator[Tuple[List[int], List[int]]]:
        """Iterate over (context, target) pairs for annealing training."""
        for sequence in self._train_sequences:
            # Random starting point
            max_start = len(sequence) - self.context_length - self.gen_length
            if max_start <= 0:
                continue
            
            start = torch.randint(0, max_start, (1,)).item()
            context = sequence[start:start + self.context_length]
            target = sequence[start + self.context_length:start + self.context_length + self.gen_length]
            
            yield context, target
    
    def _run_grammar_steps(self, num_steps: int = 5) -> None:
        """Run a fixed number of grammar steps."""
        for _ in range(num_steps):
            self.manifold.step_grammar()
    
    def train_step(self, batch: Tuple[List[int], List[int]]) -> Dict[str, float]:
        """Training step for thermodynamic text annealing.
        
        The key insight: we train by showing the manifold complete sequences,
        so it learns the structure. Then at generation time, we use that
        structure to "anneal" from noise.
        """
        context, target = batch
        full_sequence = context + target
        
        # Process the full sequence through the manifold
        seq_tensor = torch.tensor(full_sequence, device=self.device, dtype=torch.long)
        self.manifold.ingest_ids(seq_tensor)
        self._run_grammar_steps(5)
        
        # Observe each transition (this builds the bond structure)
        for i in range(len(full_sequence) - 1):
            current = full_sequence[i]
            next_token = full_sequence[i + 1]
            
            # Get current prediction
            ctx = torch.tensor(full_sequence[:i+1], device=self.device, dtype=torch.long)
            self.manifold.ingest_ids(ctx)
            self._run_grammar_steps(5)
            output = self.manifold.output_state()
            
            # Observe (learn)
            self.manifold.observe_next_token(next_token, probs=output.probs)
        
        # Compute how well we can reconstruct
        correct = 0
        for i in range(len(target)):
            ctx_len = self.context_length + i
            ctx = torch.tensor(full_sequence[:ctx_len], device=self.device, dtype=torch.long)
            self.manifold.ingest_ids(ctx)
            self._run_grammar_steps(5)
            output = self.manifold.output_state()
            
            if output.token_index == target[i]:
                correct += 1
        
        accuracy = correct / len(target)
        
        # Occasionally ponder
        if self.state.step % 20 == 0:
            self.manifold.idle_think(steps=2, dream_steps=8)
        
        return {"accuracy": accuracy}
    
    def _generate_by_annealing(
        self, 
        context: List[int],
        temperature_schedule: Optional[List[float]] = None,
    ) -> List[int]:
        """Generate text by thermodynamic annealing.
        
        1. Start with random token candidates for each position
        2. Iteratively refine using manifold dynamics
        3. Lower "temperature" over time to crystallize the sequence
        """
        if temperature_schedule is None:
            # Exponential cooling schedule
            temperature_schedule = [
                1.0 * (0.9 ** i) for i in range(self.annealing_steps)
            ]
        
        # Initialize: random tokens for generation positions
        generated = [
            torch.randint(0, len(self.vocab), (1,)).item()
            for _ in range(self.gen_length)
        ]
        
        for step, temp in enumerate(temperature_schedule):
            # For each position, consider alternatives
            for pos in range(self.gen_length):
                # Build context for this position
                prefix = context + generated[:pos]
                
                ctx_tensor = torch.tensor(prefix, device=self.device, dtype=torch.long)
                self.manifold.ingest_ids(ctx_tensor)
                
                # Run dynamics
                self._run_grammar_steps(5)
                
                output = self.manifold.output_state()
                probs = output.probs
                
                # Temperature-scaled sampling
                if temp > 0.01:
                    # Sample with temperature
                    scaled_logits = output.logits / temp
                    scaled_probs = torch.softmax(scaled_logits, dim=0)
                    
                    # Mix with current token (momentum)
                    current_prob = torch.zeros_like(scaled_probs)
                    current_prob[generated[pos]] = 0.5
                    mixed_probs = 0.5 * scaled_probs + current_prob
                    mixed_probs = mixed_probs / mixed_probs.sum()
                    
                    new_token = torch.multinomial(mixed_probs, 1).item()
                else:
                    # Greedy at low temperature
                    new_token = output.token_index
                
                generated[pos] = new_token
        
        return generated
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate generation quality."""
        total_accuracy = 0.0
        total_perplexity = 0.0
        count = 0
        
        for sequence in self._eval_sequences[:20]:
            if len(sequence) < self.context_length + self.gen_length:
                continue
            
            context = sequence[:self.context_length]
            target = sequence[self.context_length:self.context_length + self.gen_length]
            
            # Generate by annealing
            generated = self._generate_by_annealing(context)
            
            # Compare to target
            correct = sum(1 for g, t in zip(generated, target) if g == t)
            accuracy = correct / len(target)
            total_accuracy += accuracy
            
            # Compute perplexity of generated sequence
            log_prob_sum = 0.0
            full_gen = context + generated
            for i in range(len(context), len(full_gen) - 1):
                ctx = torch.tensor(full_gen[:i+1], device=self.device, dtype=torch.long)
                self.manifold.ingest_ids(ctx)
                self._think_until_complete(max_steps=20)
                output = self.manifold.output_state()
                
                next_token = full_gen[i + 1] if i + 1 < len(full_gen) else generated[-1]
                prob = output.probs[next_token].item()
                log_prob_sum += -torch.log(torch.tensor(prob + 1e-10)).item()
            
            if len(generated) > 1:
                avg_log_prob = log_prob_sum / (len(generated) - 1)
                perplexity = torch.exp(torch.tensor(avg_log_prob)).item()
                total_perplexity += perplexity
            
            count += 1
        
        if count == 0:
            return {"accuracy": 0.0, "perplexity": float("inf")}
        
        return {
            "accuracy": total_accuracy / count,
            "perplexity": total_perplexity / count,
            "samples_evaluated": count,
            "graph_edges": self.manifold.graph.num_edges,
            "chunks": self.manifold.chunks.num_chunks,
        }
    
    def generate_samples(self, num_samples: int = 5) -> List[str]:
        """Generate text samples for inspection."""
        samples = []
        
        for sequence in self._eval_sequences[:num_samples]:
            if len(sequence) < self.context_length + self.gen_length:
                continue
            
            context = sequence[:self.context_length]
            generated = self._generate_by_annealing(context)
            
            # Convert to text
            context_text = " ".join(self.vocab[i] for i in context)
            gen_text = " ".join(self.vocab[i] for i in generated)
            
            samples.append({
                "context": context_text,
                "generated": gen_text,
            })
        
        return samples


def run_text_diffusion_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = TextDiffusionExperiment(scale=scale, device=device)
    result = exp.run()
    
    # Generate some samples for inspection
    samples = exp.generate_samples(3)
    for i, sample in enumerate(samples):
        print(f"\n  Sample {i+1}:")
        print(f"    Context: ...{sample['context'][-50:]}")
        print(f"    Generated: {sample['generated']}")
    
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
        "samples": samples,
    }

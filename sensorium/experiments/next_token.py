"""Next Token Prediction Experiment

Uses WikiText-2 / WikiText-103 from HuggingFace.
Standard language modeling benchmark.

Goal: Predict next token from context.
Metrics: Accuracy, Perplexity (where applicable)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from thermo_manifold.core.diagnostics import SemanticDiagnosticsLogger
from thermo_manifold.core.viz import plot_pondering_jsonl
from thermo_manifold.semantic.hierarchical import HierarchicalSemanticManifold

from .base import BaseExperiment, Scale

DEFAULT_OUT_DIR = Path("./artifacts/next_token")


class NextTokenExperiment(BaseExperiment):
    """Next token prediction using the semantic manifold.
    
    This is the core capability of the system.
    """
    
    name = "next_token"
    goal = "Predict next token from context (language modeling)"
    
    def __init__(
        self,
        scale: Scale = Scale.TOY,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        super().__init__(scale, device, seed)
        
        # Scale-specific configs
        if scale == Scale.TOY:
            self.dataset_name = "wikitext-2-raw-v1"
            self.context_length = 32
            self.vocab_size = 1000  # Limit vocab
        elif scale == Scale.MEDIUM:
            self.dataset_name = "wikitext-2-raw-v1"
            self.context_length = 64
            self.vocab_size = 10000
        else:
            self.dataset_name = "wikitext-103-raw-v1"
            self.context_length = 128
            self.vocab_size = 50000
        
        self.vocab: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self._train_steps: int = int(self.scale_config.train_steps)
    
    def setup(self) -> None:
        """Load WikiText and build vocabulary."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"    Loading {self.dataset_name}...")
        
        dataset = load_dataset(
            "wikitext",
            self.dataset_name,
            streaming=True
        )
        
        # Build vocabulary from training data
        print(f"    Building vocabulary (max {self.vocab_size} tokens)...")
        
        word_counts: Counter = Counter()
        sample_count = 0
        max_vocab_samples = self.scale_config.max_train_samples or 10000
        
        for sample in dataset["train"]:
            text = sample["text"]
            if not text.strip():
                continue
            
            # Simple whitespace tokenization
            tokens = text.split()
            word_counts.update(tokens)
            sample_count += 1
            
            if sample_count >= max_vocab_samples:
                break
        
        # Build vocab from most common tokens
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = special_tokens + [
            word for word, _ in word_counts.most_common(self.vocab_size - len(special_tokens))
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        
        print(f"    Vocabulary size: {len(self.vocab)}")
        print(f"    Sample tokens: {self.vocab[4:14]}")
        
        # Store dataset iterators
        self.train_stream = dataset["train"]
        self.eval_stream = dataset["validation"]
        
        # Prefetch and tokenize data
        self._train_sequences: List[List[int]] = []
        self._eval_sequences: List[List[int]] = []
        
        self._tokenize_stream(
            self.train_stream,
            self._train_sequences,
            self.scale_config.max_train_samples or 10000,
        )
        self._tokenize_stream(
            self.eval_stream,
            self._eval_sequences,
            self.scale_config.max_eval_samples or 1000,
        )
        
        print(f"    Train sequences: {len(self._train_sequences)}")
        print(f"    Eval sequences: {len(self._eval_sequences)}")
        
        # Initialize semantic manifold
        self.manifold = HierarchicalSemanticManifold(
            self.physics_config,
            self.device,
            vocab=self.vocab,
            embed_dim=min(self.scale_config.embed_dim, len(self.vocab)),
            chunk_min_len=2,
            chunk_max_len=4,
        )

    def _reset_artifacts(self, out_dir: Path) -> None:
        """Ensure deterministic artifact outputs (overwrite vs append)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in [
            out_dir / "pondering.jsonl",
            out_dir / "pondering.csv",
            out_dir / "next_token_metrics.json",
            out_dir / "next_token_data.json",
            out_dir / "next_token.png",
            out_dir / "pondering.png",
            out_dir / "tables" / "next_token_summary.tex",
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # Best-effort cleanup; we still want the experiment to run.
                pass

    @staticmethod
    def _topological_entropy(src: torch.Tensor, w: torch.Tensor, eps: float) -> float:
        """Mean per-src entropy of outgoing normalized weights."""
        if src.numel() == 0:
            return 0.0
        w = w.clamp(min=0.0)
        if float(w.sum().item()) <= eps:
            return 0.0
        src_u, inv = torch.unique(src, return_inverse=True)
        out_sum = torch.zeros(int(src_u.numel()), device=w.device, dtype=w.dtype)
        out_sum.index_add_(0, inv, w)
        p = w / (out_sum[inv] + eps)
        edge_ent = -p * torch.log(p + eps)
        ent_src = torch.zeros(int(src_u.numel()), device=w.device, dtype=w.dtype)
        ent_src.index_add_(0, inv, edge_ent)
        return float(ent_src.mean().item())

    @staticmethod
    def _rolling_mean(x: torch.Tensor, win: int) -> torch.Tensor:
        win = max(1, int(win))
        if x.numel() == 0:
            return x
        kern = torch.ones(win, dtype=torch.float32, device=x.device) / float(win)
        y = torch.nn.functional.conv1d(x.view(1, 1, -1), kern.view(1, 1, -1), padding=win // 2).view(-1)
        return y[: x.numel()]

    def _generate_artifacts(self, *, out_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Path]:
        """Write figures/tables for paper artifacts."""
        figures: Dict[str, Path] = {}
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # LaTeX table (summary)
        final = metrics.get("final_eval", {}) if isinstance(metrics, dict) else {}
        acc = float(final.get("accuracy", 0.0))
        ppl = float(final.get("perplexity", float("inf")))
        edges = int(final.get("graph_edges", 0))
        chunks = int(final.get("chunks", 0))
        table = (
            r"""\begin{table}[t]
\centering
\caption{Next-token prediction after a single online training pass (no gradient optimization).}
\label{tab:next_token}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Validation accuracy & """
            + f"{acc:.1%}"
            + r""" \\
Validation perplexity & """
            + (f"{ppl:.2f}" if ppl != float("inf") else r"$\infty$")
            + r""" \\
Graph edges & """
            + f"{edges}"
            + r""" \\
Chunks & """
            + f"{chunks}"
            + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        )
        table_path = tables_dir / "next_token_summary.tex"
        table_path.write_text(table, encoding="utf-8")

        # Figures
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = int(metrics["steps"])
            x = list(range(steps))

            fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            ax[0].plot(x, metrics["acc_smooth"], label="accuracy (rolling)", linewidth=1.5)
            ax[0].set_ylabel("accuracy")
            ax[0].set_ylim(0.0, 1.05)
            ax[0].legend(loc="upper right", fontsize=8)

            ax[1].plot(x, metrics["nll_smooth"], label="NLL (rolling)", linewidth=1.2)
            ax[1].set_ylabel("NLL")
            ax[1].legend(loc="upper right", fontsize=8)

            ax[2].plot(x, metrics["energy"], label="system energy", alpha=0.7)
            ax[2].plot(x, metrics["topo_entropy"], label="topology entropy", alpha=0.7)
            ax[2].plot(x, metrics["chunks_ts"], label="#chunks", alpha=0.7)
            ax[2].set_ylabel("energy / entropy / count")
            ax[2].set_xlabel("train step")
            ax[2].legend(loc="upper right", fontsize=8)

            fig.tight_layout()
            fig_path = out_dir / "next_token.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            figures["next_token"] = fig_path
        except Exception:
            pass

        # Pondering plot from JSONL (if any)
        try:
            ponder_png = plot_pondering_jsonl(out_dir / "pondering.jsonl", out_dir / "pondering.png")
            if ponder_png.exists():
                figures["pondering"] = ponder_png
        except Exception:
            pass

        return figures

    def run(self):  # type: ignore[override]
        """Run experiment and generate paper artifacts.

        Key property: this system does *not* optimize a loss; we do not run evaluation
        during training. We pump tokens (online structural updates), then evaluate once.
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.name} ({self.scale.value})")
        print(f"Goal: {self.goal}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")

        from .base import ExperimentResult

        try:
            print("\n[1] Setup...")
            self.setup()

            out_dir = DEFAULT_OUT_DIR / self.scale.value
            self._reset_artifacts(out_dir)

            # Attach diagnostics for pondering traces (written only when idle_think is called).
            ponder_jsonl = out_dir / "pondering.jsonl"
            ponder_csv = out_dir / "pondering.csv"
            self.manifold.set_diagnostics(
                SemanticDiagnosticsLogger(csv_path=str(ponder_csv), jsonl_path=str(ponder_jsonl))
            )

            steps = int(self._train_steps)
            print(f"\n[2] Training ({steps} steps; no in-loop eval)...")

            # Per-step metrics
            acc = torch.zeros(steps, dtype=torch.float32)
            nll = torch.zeros(steps, dtype=torch.float32)
            entropy = torch.zeros(steps, dtype=torch.float32)
            energy = torch.zeros(steps, dtype=torch.float32)
            topo = torch.zeros(steps, dtype=torch.float32)
            chunks_ts = torch.zeros(steps, dtype=torch.float32)

            data_iter = self.train_iterator()
            for t in range(steps):
                try:
                    context, target = next(data_iter)
                except StopIteration:
                    # Restart streaming iterator if it ended early.
                    data_iter = self.train_iterator()
                    context, target = next(data_iter)

                context_tensor = torch.tensor(context, device=self.device, dtype=torch.long)
                self.manifold.ingest_ids(context_tensor)

                # Fixed steps for speed (dt-calibrated heuristic)
                num_steps = 3 if self.scale == Scale.TOY else 5
                for _ in range(num_steps):
                    self.manifold.step_grammar()

                out = self.manifold.output_state()
                pred = int(out.token_index)
                acc[t] = 1.0 if pred == int(target) else 0.0

                p = float(out.probs[int(target)].clamp(min=1e-10).item())
                nll[t] = float((-torch.log(torch.tensor(p, dtype=torch.float32))).item())
                entropy[t] = float(out.meta.get("entropy", 0.0))

                # System metrics (exclude long-term structural mass where possible)
                _exc = self.manifold.attractors.get("excitation").abs().sum()
                _heat = self.manifold.attractors.get("heat").abs().sum()
                _cexc = self.manifold.chunks.excitation.abs().sum()
                _cheat = self.manifold.chunks.heat.abs().sum()
                energy[t] = (_exc + _heat + _cexc + _cheat).detach().to(torch.float32).cpu()
                topo[t] = torch.tensor(
                    self._topological_entropy(self.manifold.graph.src.detach(), self.manifold.graph.w.detach(), eps=self.physics_config.eps),
                    dtype=torch.float32,
                )
                chunks_ts[t] = float(self.manifold.chunks.num_chunks)

                # Online structural update ("learning")
                self.manifold.observe_next_token(int(target), probs=out.probs)

                # Idle pondering cadence (deterministic; writes diagnostics via logger)
                if t > 0 and (t % 50 == 0):
                    self.manifold.idle_think(steps=1, dream_steps=2)

            # Smooth curves for readability
            acc_smooth = self._rolling_mean(acc, win=200).cpu().tolist()
            nll_smooth = self._rolling_mean(nll, win=200).cpu().tolist()

            # Save training time series + config
            metrics: Dict[str, Any] = {
                "config": asdict(self.physics_config),
                "scale": self.scale.value,
                "dataset": self.dataset_name,
                "context_length": int(self.context_length),
                "vocab_size": int(len(self.vocab)),
                "steps": steps,
                "acc": acc.cpu().tolist(),
                "acc_smooth": acc_smooth,
                "nll": nll.cpu().tolist(),
                "nll_smooth": nll_smooth,
                "entropy": entropy.cpu().tolist(),
                "energy": energy.cpu().tolist(),
                "topo_entropy": topo.cpu().tolist(),
                "chunks_ts": chunks_ts.cpu().tolist(),
            }

            (out_dir / "next_token_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            print("\n[3] Final evaluation...")
            final_eval = self.evaluate()
            metrics["final_eval"] = final_eval

            # Save a compact summary JSON (avoids duplicating full time series)
            summary = {k: v for k, v in metrics.items() if not isinstance(v, list)}
            (out_dir / "next_token_data.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

            figures = self._generate_artifacts(out_dir=out_dir, metrics=metrics)

            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=True,
                metrics={"final": final_eval, "artifacts_dir": str(out_dir)},
                tables={"next_token_summary": (out_dir / "tables" / "next_token_summary.tex").read_text(encoding="utf-8")},
                figures=figures,
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"\n[FAILED] {e}")
            print(tb)
            return ExperimentResult(
                name=self.name,
                scale=self.scale.value,
                goal=self.goal,
                success=False,
                metrics={},
                failure_reason=str(e),
            )
    
    def _tokenize_stream(
        self,
        stream,
        output: List[List[int]],
        max_samples: int,
    ) -> None:
        """Tokenize a stream into sequences."""
        unk_id = self.token_to_id.get("<unk>", 1)
        
        for sample in stream:
            text = sample["text"]
            if not text.strip():
                continue
            
            tokens = text.split()
            if len(tokens) < 2:
                continue
            
            # Convert to IDs
            ids = [self.token_to_id.get(t, unk_id) for t in tokens]
            output.append(ids)
            
            if len(output) >= max_samples:
                break
    
    def train_iterator(self) -> Iterator[Tuple[List[int], int]]:
        """Iterate over training (context, next_token) pairs."""
        for sequence in self._train_sequences:
            # Sliding window
            for i in range(len(sequence) - 1):
                start = max(0, i - self.context_length + 1)
                context = sequence[start:i+1]
                target = sequence[i + 1]
                yield context, target
    
    def _run_grammar_steps(self, num_steps: int = 5) -> None:
        """Run a fixed number of grammar steps.
        
        Note: thinking_complete() can be slow to converge, so for experiments
        we use a fixed number of steps calibrated to dt.
        """
        for _ in range(num_steps):
            self.manifold.step_grammar()
    
    def train_step(self, batch: Tuple[List[int], int]) -> Dict[str, float]:
        """One step of thermodynamic language modeling."""
        context, target = batch
        
        # Convert to tensor
        context_tensor = torch.tensor(context, device=self.device, dtype=torch.long)
        
        # Ingest context
        self.manifold.ingest_ids(context_tensor)
        
        # Run grammar dynamics (fixed steps for speed)
        # Toy scale benefits from fewer steps on CPU.
        num_steps = 3 if self.scale == Scale.TOY else 5
        self._run_grammar_steps(num_steps=num_steps)
        
        # Get prediction
        output = self.manifold.output_state()
        predicted = output.token_index
        
        # Observe actual next token (this is the learning!)
        probs = output.probs
        self.manifold.observe_next_token(target, probs=probs)
        
        # Optionally do some pondering
        # NOTE: idle_think/dreaming is comparatively expensive; avoid doing it on
        # step 0 (it makes the progress bar look "stuck") and run it less often.
        if self.state.step > 0 and self.state.step % 50 == 0:
            self.manifold.idle_think(steps=1, dream_steps=2)
        
        # Metrics
        correct = 1.0 if predicted == target else 0.0
        
        # Compute approximate perplexity contribution
        prob = probs[target].item()
        log_prob = -torch.log(torch.tensor(prob + 1e-10)).item()
        
        return {
            "accuracy": correct,
            "log_prob": log_prob,
            "entropy": output.meta.get("entropy", 0.0),
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        correct = 0
        total = 0
        total_log_prob = 0.0
        
        # Note: evaluation is run once after the online training pass.
        # `_eval_sequences` is already bounded during prefetch/tokenization by scale.
        for sequence in self._eval_sequences:
            for i in range(len(sequence) - 1):
                start = max(0, i - self.context_length + 1)
                context = sequence[start:i+1]
                target = sequence[i + 1]
                
                context_tensor = torch.tensor(context, device=self.device, dtype=torch.long)
                self.manifold.ingest_ids(context_tensor)
                num_steps = 3 if self.scale == Scale.TOY else 5
                self._run_grammar_steps(num_steps=num_steps)
                
                output = self.manifold.output_state()
                predicted = output.token_index
                
                if predicted == target:
                    correct += 1
                
                prob = output.probs[target].item()
                total_log_prob += -torch.log(torch.tensor(prob + 1e-10)).item()
                
                total += 1
                
                # Don't learn during eval
        
        if total == 0:
            return {"accuracy": 0.0, "perplexity": float("inf")}
        
        avg_log_prob = total_log_prob / total
        perplexity = torch.exp(torch.tensor(avg_log_prob)).item()
        
        return {
            "accuracy": correct / total,
            "perplexity": perplexity,
            "eval_tokens": total,
            "graph_edges": self.manifold.graph.num_edges,
            "chunks": self.manifold.chunks.num_chunks,
        }


def run_next_token_experiment(
    scale: Scale = Scale.TOY,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to run the experiment."""
    exp = NextTokenExperiment(scale=scale, device=device)
    result = exp.run()
    return {
        "result": result,
        "success": result.success,
        "metrics": result.metrics,
    }

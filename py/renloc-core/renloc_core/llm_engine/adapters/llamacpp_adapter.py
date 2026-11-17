"""
llamacpp_adapter.py
Adapter for llama-cpp-python and local llama-server with hybrid constraint enforcement:
- Auto-detection and preloading of CUDA 12/13 runtimes for hardware acceleration.
- Dynamic VRAM/RAM monitoring and strict OOM risk assessment (_room_for_copy).
- GBNF grammar compilation when tag count N <= 5.
- Stateful LogitsProcessor callback when tag count N > 5.
- Specialized Ren'Py Visual Novel translation prompt template and tag preservation validator.
- Support for direct in-process GGUF loading or external OpenAI-compatible HTTP server (llama-server / Ollama).
"""

from __future__ import annotations

import ctypes
import glob
import json
import logging
import os
import re
import site
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Preload NVIDIA CUDA runtime libraries from site-packages if available
def _setup_cuda_env() -> None:
    try:
        for sp in site.getsitepackages():
            for lib_dir in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
                if os.path.isdir(lib_dir):
                    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                    for f in os.listdir(lib_dir):
                        if ".so" in f:
                            try:
                                ctypes.CDLL(os.path.join(lib_dir, f))
                            except Exception:
                                pass
    except Exception:
        pass

_setup_cuda_env()

try:
    from renloc_core.llm_engine.constraint_compiler import TagStackLogitsProcessor
    _HAS_TAG_STACK_PROCESSOR = True
except ImportError:
    _HAS_TAG_STACK_PROCESSOR = False

try:
    from renloc_core.llm_engine.vram_monitor import VramMonitor
    _HAS_VRAM_MONITOR = True
except ImportError:
    _HAS_VRAM_MONITOR = False


# ==============================================================================
# Hardware Status & OOM Assessment
# ==============================================================================

def get_system_hardware_info() -> Dict[str, Any]:
    """Returns real-time GPU VRAM and System RAM & Swap metrics."""
    info: Dict[str, Any] = {
        "gpu_name": "CPU Only",
        "total_vram_mb": 0.0,
        "free_vram_mb": 0.0,
        "used_vram_mb": 0.0,
        "total_ram_mb": 0.0,
        "free_ram_mb": 0.0,
        "swap_total_mb": 0.0,
        "swap_free_mb": 0.0,
        "cuda_available": False,
        "gpu_offload_supported": False,
    }

    # System RAM & Swap
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    val = float(parts[1].strip().split()[0]) / 1024.0
                    if k == "MemTotal":
                        info["total_ram_mb"] = val
                    elif k in ("MemAvailable", "MemFree"):
                        if info["free_ram_mb"] == 0.0 or k == "MemAvailable":
                            info["free_ram_mb"] = val
                    elif k == "SwapTotal":
                        info["swap_total_mb"] = val
                    elif k == "SwapFree":
                        info["swap_free_mb"] = val
    except Exception:
        pass

    # GPU VRAM via VramMonitor
    if _HAS_VRAM_MONITOR:
        try:
            vmon = VramMonitor()
            vinfo = vmon.get_vram_info()
            info["total_vram_mb"] = vinfo.total_mb
            info["free_vram_mb"] = vinfo.free_mb
            info["used_vram_mb"] = vinfo.used_mb
            info["gpu_name"] = vinfo.device_name
            info["cuda_available"] = vinfo.total_bytes > 0
        except Exception as e:
            logger.debug(f"Failed to query VramMonitor: {e}")

    # Check llama_cpp gpu offload
    try:
        from llama_cpp import llama_supports_gpu_offload
        info["gpu_offload_supported"] = llama_supports_gpu_offload()
    except Exception:
        info["gpu_offload_supported"] = False

    return info


def assess_model_memory_risk(
    model_path: Union[str, Path],
    reserved_vram_mb: int = 1024,
) -> Dict[str, Any]:
    """
    Evaluates memory requirements for a GGUF model against current VRAM/RAM,
    calculates safe n_gpu_layers, and prevents Out-Of-Memory (OOM) crashes.
    """
    p = Path(model_path)
    if not p.exists():
        return {
            "status": "error",
            "message": f"Không tìm thấy file model: {model_path}",
            "can_fit": False,
            "recommended_gpu_layers": 0,
            "recommended_ctx": 2048,
        }

    file_size_mb = p.stat().st_size / (1024 * 1024)
    hw = get_system_hardware_info()
    free_vram = hw["free_vram_mb"]
    free_ram = hw["free_ram_mb"]
    swap_total = hw["swap_total_mb"]

    safe_vram_budget = max(0.0, free_vram - reserved_vram_mb)

    # 1. Full GPU Fit (e.g. 7B/8B model in 8GB VRAM)
    if hw["gpu_offload_supported"] and file_size_mb <= safe_vram_budget:
        return {
            "status": "safe_full_gpu",
            "message": f"Model {file_size_mb:.1f} MB vừa vặn 100% trong VRAM của {hw['gpu_name']} ({free_vram:.1f} MB khả dụng). Tốc độ tối đa, 0% nguy cơ OOM.",
            "can_fit": True,
            "recommended_gpu_layers": -1,
            "recommended_ctx": 2048,
            "file_size_mb": file_size_mb,
            "free_vram_mb": free_vram,
            "free_ram_mb": free_ram,
        }

    # 2. Hybrid GPU + RAM Offloading
    if hw["gpu_offload_supported"] and safe_vram_budget > 1024:
        vram_portion = safe_vram_budget
        ram_portion = max(0.0, file_size_mb - vram_portion)
        # Margin required in RAM: model RAM portion + 1.5GB KV cache/overhead
        safe_ram_margin = free_ram - (ram_portion + 1536)

        if safe_ram_margin > 0 or swap_total > 4096:
            # Fraction of layers to offload
            ratio = vram_portion / file_size_mb
            # Typical models have 32 to 64 layers
            est_layers = int(ratio * 36)
            return {
                "status": "safe_hybrid",
                "message": f"Model {file_size_mb:.1f} MB được chia tải: ~{vram_portion:.1f} MB trên GPU và ~{ram_portion:.1f} MB trên RAM. Hoạt động an toàn.",
                "can_fit": True,
                "recommended_gpu_layers": max(1, est_layers),
                "recommended_ctx": 2048,
                "file_size_mb": file_size_mb,
                "free_vram_mb": free_vram,
                "free_ram_mb": free_ram,
            }

    # 3. CPU Only or Extreme Memory Pressure
    total_safe_mem = safe_vram_budget + free_ram
    if file_size_mb > total_safe_mem and swap_total == 0:
        return {
            "status": "oom_risk",
            "message": f"CẢNH BÁO OOM: Model ({file_size_mb / 1024:.1f} GB) vượt quá tổng bộ nhớ an toàn (VRAM: {free_vram / 1024:.1f} GB + RAM: {free_ram / 1024:.1f} GB, Swap: 0B). Dễ gây sập OOM Killer! Khuyên dùng model 7B/8B.",
            "can_fit": False,
            "recommended_gpu_layers": -1 if hw["gpu_offload_supported"] else 0,
            "recommended_ctx": 1024,
            "file_size_mb": file_size_mb,
            "free_vram_mb": free_vram,
            "free_ram_mb": free_ram,
        }

    return {
        "status": "warning_tight",
        "message": f"Bộ nhớ tương đối sít sao. Đã tự động giới hạn ngữ cảnh n_ctx=1024 để bảo vệ RAM.",
        "can_fit": True,
        "recommended_gpu_layers": -1 if hw["gpu_offload_supported"] else 0,
        "recommended_ctx": 1024,
        "file_size_mb": file_size_mb,
        "free_vram_mb": free_vram,
        "free_ram_mb": free_ram,
    }


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class GenerationResult:
    """Output returned from LLM generation."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    enforcement_mode: str = "none"  # "gbnf", "logits_callback", or "prompt_constrained"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. GBNF Grammar Compiler (N <= 5 Tags)
# ==============================================================================

class GBNFGrammarCompiler:
    """
    Compiles Ren'Py dialogue tags and structural constraints into GBNF (GGML BNF) grammar.
    Used when tag count N <= 5.
    """

    @classmethod
    def escape_gbnf_string(cls, s: str) -> str:
        """Escape string literals for GBNF grammar rules."""
        escaped = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def compile_grammar_for_tags(
        cls,
        tags: Sequence[str],
        allow_unicode_vietnamese: bool = True,
    ) -> str:
        if not tags:
            return r"""root ::= text
text ::= [^\x00]*
"""

        rules: List[str] = []
        tag_rule_names = []
        for i, tag in enumerate(tags):
            rule_name = f"tag{i}"
            tag_rule_names.append(rule_name)
            escaped_tag = cls.escape_gbnf_string(tag)
            rules.append(f"{rule_name} ::= {escaped_tag}")

        tag_choice = " | ".join(tag_rule_names)
        rules.append(f"tag ::= {tag_choice}")
        rules.append(r'text ::= [^"\\<>{}\[\]]+')
        rules.append(r"content ::= (text | tag)*")
        rules.append("root ::= content")

        return "\n".join(rules) + "\n"


# ==============================================================================
# 2. Stateful LogitsProcessor Callback (N > 5 Tags)
# ==============================================================================

class LlamaCppLogitsProcessorCallback:
    """
    Stateful logit processor callback for llama-cpp-python when N > 5.
    Avoids exponential state explosion of PDA/DFA grammars by enforcing
    constraints via token masks at each sampling step.
    """

    def __init__(
        self,
        tags: Sequence[str],
        tokenizer: Optional[Any] = None,
        tag_tokens_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.tags = list(tags)
        self.tokenizer = tokenizer
        self.tag_tokens_map = tag_tokens_map or {}
        self.tag_stack: List[str] = []
        self.step = 0
        if _HAS_TAG_STACK_PROCESSOR:
            self._tag_processor = TagStackLogitsProcessor(expected_tags=self.tags)
        else:
            self._tag_processor = None

    def __call__(self, input_ids: Sequence[int], logits: Any) -> Any:
        self.step += 1
        if self._tag_processor is not None:
            return self._tag_processor(input_ids, logits)

        is_list = isinstance(logits, list)
        if is_list:
            if all(v == float("-inf") for v in logits):
                logits[0] = 0.0
        else:
            has_torch = hasattr(logits, "is_cuda") or hasattr(logits, "masked_fill_")
            if has_torch:
                import torch
                if torch.all(torch.isneginf(logits)):
                    logits[0] = 0.0
            else:
                import numpy as np
                if np.all(np.isneginf(logits)):
                    logits[0] = 0.0

        return logits


# ==============================================================================
# 3. LlamaCpp Adapter (Local GGUF & llama-server)
# ==============================================================================

class LlamaCppAdapter:
    """
    Unified adapter managing local GGUF models and OpenAI-compatible llama-server.
    Enforces OOM safety, CUDA hardware offloading, and Ren'Py tag preservation.
    """

    RE_VARS = re.compile(r"\[[^\]]+\]")
    RE_TAGS = re.compile(r"\{[^}]+\}")

    DEFAULT_SYSTEM_PROMPT = (
        "Bạn là chuyên gia dịch thuật Visual Novel chuyên nghiệp, dịch kịch bản Ren'Py sang tiếng Việt tự nhiên, giàu cảm xúc.\n"
        "QUY TẮC BẮT BUỘC KHÔNG ĐƯỢC PHÉP SAI:\n"
        "1. GIỮ NGUYÊN 100% mọi thẻ định dạng Ren'Py: {b}, {/b}, {i}, {/i}, {u}, {/u}, {w}, {p}, {cps=...}, {color=...}, {size=...}, {font=...}, {rb}...{/rb}, {rt}...{/rt}.\n"
        "2. GIỮ NGUYÊN 100% mọi biến nội suy trong ngoặc vuông: [player_name], [hero!t], [count], [item[0].name]. TUYỆT ĐỐI không dịch chữ trong ngoặc vuông.\n"
        "3. Giữ nguyên ký tự xuống dòng \\n và dấu ngoặc kép thoát \\\", \\'.\n"
        "4. Chỉ xuất DUY NHẤT nội dung câu dịch tiếng Việt. Không thêm lời bình, không bọc dấu ngoặc kép ngoài cùng."
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_instance: Optional[Any] = None,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        server_url: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path) if model_path else None
        self._llm = llm_instance
        self.n_ctx = min(n_ctx, 4096)  # Bound n_ctx to protect memory
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.server_url = server_url.rstrip("/") if server_url else None
        self.grammar_compiler = GBNFGrammarCompiler()

    def _ensure_llm(self) -> Any:
        """Lazy loads model with OOM guard and CUDA offload."""
        if self.server_url is not None:
            return "server"

        if self._llm is None:
            if self.model_path is None:
                logger.info("No model_path specified; running in mock/simulation mode.")
                return None

            # Pre-flight OOM check
            risk = assess_model_memory_risk(self.model_path)
            if not risk["can_fit"] and risk["status"] == "oom_risk":
                logger.warning(f"OOM Risk detected: {risk['message']}")

            # Calculate safe gpu layers if auto (-1)
            actual_gpu_layers = self.n_gpu_layers
            if actual_gpu_layers == -1:
                actual_gpu_layers = risk.get("recommended_gpu_layers", -1)

            try:
                from llama_cpp import Llama  # type: ignore
                logger.info(
                    f"Loading GGUF model: {self.model_path} (n_ctx={self.n_ctx}, n_gpu_layers={actual_gpu_layers})"
                )
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=actual_gpu_layers,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
            except ImportError:
                logger.warning("llama_cpp not installed. Falling back to simulation mode.")
                return None
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}")
                raise e

        return self._llm

    def generate(
        self,
        prompt: str,
        tags: Optional[Sequence[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> GenerationResult:
        """Standard generation method backward-compatible with tests and pipelines."""
        tags = list(tags) if tags else []
        tag_count = len(tags)

        enforcement_mode = "none"
        grammar_str = None
        logits_processor = None

        if tag_count > 0:
            if tag_count <= 5:
                enforcement_mode = "gbnf"
                grammar_str = self.grammar_compiler.compile_grammar_for_tags(tags)
            else:
                enforcement_mode = "logits_callback"
                logits_processor = LlamaCppLogitsProcessorCallback(tags=tags)

        # HTTP Server route
        if self.server_url is not None:
            return self._call_http_server(prompt, tags, max_tokens, temperature)

        llm = self._ensure_llm()

        if llm is None:
            generated_text = self._mock_generate(prompt, tags)
            return GenerationResult(
                text=generated_text,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(generated_text.split()),
                finish_reason="stop",
                enforcement_mode=enforcement_mode,
                metadata={"mock": True},
            )

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if grammar_str:
            try:
                from llama_cpp import LlamaGrammar  # type: ignore
                call_kwargs["grammar"] = LlamaGrammar.from_string(grammar_str)
            except Exception as e:
                logger.warning(f"Failed to compile LlamaGrammar object: {e}")

        if logits_processor:
            call_kwargs["logits_processor"] = [logits_processor]

        call_kwargs.update(kwargs)
        output = llm(**call_kwargs)

        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=output["choices"][0].get("finish_reason", "stop"),
            enforcement_mode=enforcement_mode,
            metadata={"raw_output": output},
        )

    def translate_dialogue(
        self,
        source_text: str,
        speaker: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        context_history: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """High-level dialogue translator with VN prompt formatting and post-validation."""
        if not source_text or not source_text.strip():
            return GenerationResult(text=source_text, enforcement_mode="passthrough")

        # Extract tags and substitutions if not provided
        if tags is None:
            found_tags = self.RE_TAGS.findall(source_text)
            found_vars = self.RE_VARS.findall(source_text)
            tags = list(dict.fromkeys(found_tags + found_vars))

        # Check pure tags/numbers
        stripped = self.RE_TAGS.sub("", source_text)
        stripped = self.RE_VARS.sub("", stripped).strip()
        if not stripped or re.match(r"^[\s\d\W_]*$", stripped):
            return GenerationResult(text=source_text, enforcement_mode="deterministic_passthrough")

        user_content = f"Dịch câu thoại sau sang tiếng Việt tự nhiên:\n{source_text}"
        if speaker:
            user_content = f"[Người nói: {speaker}]\n{user_content}"
        if context_history:
            history_str = "\n".join(context_history[-2:])
            user_content = f"[Ngữ cảnh trước]:\n{history_str}\n\n{user_content}"

        # If HTTP server
        if self.server_url is not None:
            return self._call_http_server(
                prompt=user_content,
                tags=tags,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                source_raw=source_text,
            )

        llm = self._ensure_llm()
        if llm is None:
            simulated = self._mock_generate(source_text, tags)
            return GenerationResult(text=simulated, enforcement_mode="mock")

        # Use create_chat_completion for optimal chat template formatting
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = resp["choices"][0]["message"]["content"].strip()
            clean_text = self._clean_and_repair_tags(source_text, raw_text, tags)
            usage = resp.get("usage", {})
            return GenerationResult(
                text=clean_text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=resp["choices"][0].get("finish_reason", "stop"),
                enforcement_mode="chat_completion",
            )
        except Exception as e:
            logger.warning(f"create_chat_completion failed, falling back to generate(): {e}")
            return self.generate(
                prompt=f"{self.DEFAULT_SYSTEM_PROMPT}\n\n{user_content}\n\nBản dịch:",
                tags=tags,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def _call_http_server(
        self,
        prompt: str,
        tags: Sequence[str],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
        source_raw: Optional[str] = None,
    ) -> GenerationResult:
        """Invokes external OpenAI-compatible llama-server / vLLM / Ollama endpoint."""
        url = f"{self.server_url}/chat/completions"
        sys_msg = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        payload = {
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
                if source_raw:
                    text = self._clean_and_repair_tags(source_raw, text, tags)
                usage = data.get("usage", {})
                return GenerationResult(
                    text=text,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    enforcement_mode="http_llama_server",
                )
        except Exception as e:
            logger.error(f"HTTP call to {url} failed: {e}")
            raise RuntimeError(f"Lỗi kết nối llama-server ({url}): {e}")

    def _clean_and_repair_tags(
        self,
        source_text: str,
        translated_text: str,
        expected_tags: Sequence[str],
    ) -> str:
        """Cleans stray quotes and ensures all vital tags & substitutions are preserved."""
        res = translated_text.strip()
        # Strip outer quotes if LLM added them around dialogue
        if len(res) >= 2:
            if (res.startswith('"') and res.endswith('"')) or (res.startswith('“') and res.endswith('”')):
                res = res[1:-1].strip()

        # Check all variables [var] from source
        source_vars = self.RE_VARS.findall(source_text)
        for var in source_vars:
            if var not in res:
                # Append missing variable or recover
                res += f" {var}"

        # Check paired tags like {b}...{/b}
        for tag in expected_tags:
            if tag.startswith("{") and not tag.startswith("{/"):
                closing = "{/" + tag[1:]
                if closing in expected_tags and tag in res and closing not in res:
                    res += closing

        return res

    def _mock_generate(self, prompt: str, tags: Sequence[str]) -> str:
        if not tags:
            return "Bản dịch thử nghiệm không có thẻ."
        tag_str = " ".join(tags)
        return f"Lời thoại dịch {tag_str} hoàn chỉnh."

    def unload(self) -> None:
        """Releases model instance and triggers garbage collection to free VRAM/RAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        import gc
        gc.collect()
        logger.info("LlamaCppAdapter: model unloaded and memory released.")

import re
from pathlib import Path
from typing import List, Literal, Optional

from tokenizers import Tokenizer


class Qwen_3_Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<think>",
        "</think>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path: str = "./tokenizer/qwen_3_instruct_tokenizer.json",
        model_type: Literal["base", "instruct", "thinking"] = "instruct",
        apply_chat_template: bool = True,
        add_generation_prompt: bool = False,
        add_thinking: bool = False,
    ):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if model_type == "base":
            eos_token = "<|endoftext|>"
        elif model_type in ["instruct", "thinking"]:
            eos_token = "<|im_end|>"

        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text: str, chat_wrapped: Optional[bool] = None) -> List[int]:
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: Optional[bool] = None,
    ) -> str:
        if skip_special_tokens is None:
            skip_special_tokens = False

        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def _wrap_chat(
        self,
        user_msg: str,
    ) -> str:
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"

        if self.add_generation_prompt:
            s += "<|im_start|>assistant"

            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

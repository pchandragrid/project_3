import argparse
import os
from typing import Tuple

import onnx
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class BlipEncoderWrapper(torch.nn.Module):
    def __init__(self, model: BlipForConditionalGeneration) -> None:
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state


class BlipDecoderWrapper(torch.nn.Module):
    def __init__(self, model: BlipForConditionalGeneration) -> None:
        super().__init__()
        self.text_decoder = model.text_decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2], device=encoder_hidden_states.device, dtype=attention_mask.dtype
        )
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits


def get_start_token_id(processor: BlipProcessor, model: BlipForConditionalGeneration) -> int:
    tokenizer = processor.tokenizer
    text_config = getattr(model.config, "text_config", None)
    for candidate in (
        getattr(model.config, "decoder_start_token_id", None),
        getattr(text_config, "decoder_start_token_id", None),
        getattr(text_config, "bos_token_id", None),
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    ):
        if candidate is not None:
            return int(candidate)
    return 0


def get_encoder_seq_len(model: BlipForConditionalGeneration, image_size: int) -> int:
    vision_cfg = getattr(model.config, "vision_config", None)
    patch_size = getattr(vision_cfg, "patch_size", 16) if vision_cfg is not None else 16
    return (image_size // patch_size) ** 2 + 1


def export_onnx(checkpoint_dir: str, output_dir: str, image_size: int, opset: int) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
    processor = BlipProcessor.from_pretrained(checkpoint_dir)
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    encoder_wrapper = BlipEncoderWrapper(model).eval()
    decoder_wrapper = BlipDecoderWrapper(model).eval()

    encoder_path = os.path.join(output_dir, "blip_encoder.onnx")
    decoder_path = os.path.join(output_dir, "blip_decoder.onnx")

    dummy_pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    start_token_id = get_start_token_id(processor, model)
    dummy_input_ids = torch.tensor([[start_token_id]], dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    hidden_size = model.config.text_config.hidden_size
    encoder_seq_len = get_encoder_seq_len(model, image_size)
    dummy_encoder_states = torch.randn(1, encoder_seq_len, hidden_size, dtype=torch.float32)

    torch.onnx.export(
        encoder_wrapper,
        (dummy_pixel_values,),
        encoder_path,
        export_params=True,
        input_names=["pixel_values"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_encoder_states),
        decoder_path,
        export_params=True,
        input_names=["input_ids", "attention_mask", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(encoder_path))
    onnx.checker.check_model(onnx.load(decoder_path))
    return encoder_path, decoder_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BLIP encoder and decoder to ONNX.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--output_dir", type=str, default="tasks/task1_blip_optimization/exports/onnx")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    encoder_path, decoder_path = export_onnx(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        opset=args.opset,
    )
    print(f"Encoder ONNX: {encoder_path}")
    print(f"Decoder ONNX: {decoder_path}")


if __name__ == "__main__":
    main()


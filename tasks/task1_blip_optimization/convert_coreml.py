import argparse
import importlib.util
import os
import sys

import coremltools as ct
import numpy as np
import torch
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
from transformers import BlipForConditionalGeneration


def validate_coreml_runtime() -> None:
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "CoreML conversion is not supported on this Python version. "
            "Create a Python 3.10-3.12 environment for conversion."
        )

    if importlib.util.find_spec("coremltools.libcoremlpython") is None:
        raise RuntimeError(
            "coremltools native runtime is unavailable (`coremltools.libcoremlpython` missing). "
            "Use Python 3.10-3.12 on macOS arm64 and reinstall coremltools."
        )


def patch_coremltools_int_cast_bug() -> None:
    """
    Work around a known CoreMLTools torch frontend bug where `_cast` does
    `int(x.val)` for const numpy arrays of shape (1,), which raises:
    TypeError: only 0-dimensional arrays can be converted to Python scalars
    """
    try:
        from coremltools.converters.mil.frontend.torch import ops as torch_ops
    except Exception:
        return

    original_cast = getattr(torch_ops, "_cast", None)
    if original_cast is None:
        return

    if getattr(torch_ops, "_patched_scalar_cast", False):
        return

    def _cast_safe(context, node, dtype, dtype_name):
        try:
            return original_cast(context, node, dtype, dtype_name)
        except TypeError as exc:
            if "0-dimensional arrays" not in str(exc):
                raise

            inputs = torch_ops._get_inputs(context, node, expected=1)
            x = inputs[0]
            if not x.can_be_folded_to_const():
                raise

            raw_val = np.asarray(x.val)
            if raw_val.size == 1:
                scalar_val = raw_val.reshape(()).item()
                converted = dtype(scalar_val)
            else:
                if dtype_name == "int32":
                    converted = raw_val.astype(np.int32)
                elif dtype_name == "bool":
                    converted = raw_val.astype(np.bool_)
                else:
                    converted = raw_val.astype(raw_val.dtype)

            res = torch_ops.mb.const(val=converted, name=node.name)
            context.add(res, node.name)

    torch_ops._cast = _cast_safe
    torch_ops._patched_scalar_cast = True


class BlipEncoderWrapper(torch.nn.Module):
    def __init__(self, model: BlipForConditionalGeneration) -> None:
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        return outputs.last_hidden_state


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
        # CoreML commonly feeds int32; cast inside wrapper for HF decoder.
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            device=encoder_hidden_states.device,
            dtype=torch.long,
        )
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits


def get_encoder_seq_len(model: BlipForConditionalGeneration, image_size: int) -> int:
    vision_cfg = getattr(model.config, "vision_config", None)
    patch_size = getattr(vision_cfg, "patch_size", 16) if vision_cfg is not None else 16
    return (image_size // patch_size) ** 2 + 1


def convert_from_onnx(onnx_path: str, compute_units: str):
    units = getattr(ct.ComputeUnit, compute_units)
    try:
        from onnx_coreml import convert as onnx_convert

        return onnx_convert(
            model=onnx_path,
            minimum_ios_deployment_target="15",
            compute_units=units,
        )
    except Exception:
        pass

    onnx_converter = getattr(ct.converters, "onnx", None)
    if onnx_converter is not None and hasattr(onnx_converter, "convert"):
        return onnx_converter.convert(model=onnx_path, compute_units=units)

    raise RuntimeError(
        "No ONNX -> CoreML converter available. Install `onnx-coreml` "
        "or use a coremltools build that exposes `ct.converters.onnx.convert`."
    )


def convert_from_torch(checkpoint_dir: str, output_dir: str, compute_units: str, image_size: int, max_seq_len: int) -> None:
    units = getattr(ct.ComputeUnit, compute_units)
    patch_coremltools_int_cast_bug()

    model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    os.makedirs(output_dir, exist_ok=True)
    encoder_mlpackage = os.path.join(output_dir, "blip_encoder.mlpackage")
    decoder_mlpackage = os.path.join(output_dir, "blip_decoder.mlpackage")
    encoder_q4_mlpackage = os.path.join(output_dir, "blip_encoder_q4.mlpackage")
    decoder_q4_mlpackage = os.path.join(output_dir, "blip_decoder_q4.mlpackage")

    encoder_wrapper = BlipEncoderWrapper(model).eval()
    hidden_size = model.config.text_config.hidden_size
    encoder_seq_len = get_encoder_seq_len(model, image_size)

    encoder_dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    traced_encoder = torch.jit.trace(encoder_wrapper, (encoder_dummy,), strict=False)
    encoder_mlmodel = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        compute_units=units,
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=(1, 3, image_size, image_size),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="encoder_hidden_states", dtype=np.float32)],
    )
    encoder_mlmodel.save(encoder_mlpackage)
    quantize_to_4bit(encoder_mlmodel).save(encoder_q4_mlpackage)

    decoder_wrapper = BlipDecoderWrapper(model).eval()
    ids_dummy = torch.ones((1, max_seq_len), dtype=torch.int32)
    mask_dummy = torch.ones((1, max_seq_len), dtype=torch.int32)
    enc_states_dummy = torch.randn(1, encoder_seq_len, hidden_size, dtype=torch.float32)
    traced_decoder = torch.jit.trace(
        decoder_wrapper,
        (ids_dummy, mask_dummy, enc_states_dummy),
        strict=False,
    )
    decoder_mlmodel = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        compute_units=units,
        minimum_deployment_target=ct.target.iOS15,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, ct.RangeDim(1, max_seq_len)),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, ct.RangeDim(1, max_seq_len)),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="encoder_hidden_states",
                shape=(1, encoder_seq_len, hidden_size),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    decoder_mlmodel.save(decoder_mlpackage)
    quantize_to_4bit(decoder_mlmodel).save(decoder_q4_mlpackage)


def quantize_to_4bit(mlmodel):
    config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
    return palettize_weights(mlmodel, config=config)


def convert_file(onnx_path: str, output_path: str, quantized_output_path: str, compute_units: str) -> None:
    mlmodel = convert_from_onnx(onnx_path=onnx_path, compute_units=compute_units)
    mlmodel.save(output_path)
    q_model = quantize_to_4bit(mlmodel)
    q_model.save(quantized_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX BLIP encoder/decoder to CoreML and 4-bit quantize.")
    parser.add_argument("--onnx_dir", type=str, default="tasks/task1_blip_optimization/exports/onnx")
    parser.add_argument("--output_dir", type=str, default="tasks/task1_blip_optimization/exports/coreml")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--compute_units", type=str, default="CPU_AND_NE")
    parser.add_argument("--conversion_mode", type=str, default="auto", choices=["auto", "onnx", "torch"])
    parser.add_argument("--encoder_onnx", type=str, default=None)
    parser.add_argument("--decoder_onnx", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_seq_len", type=int, default=40)
    args = parser.parse_args()
    validate_coreml_runtime()

    os.makedirs(args.output_dir, exist_ok=True)
    encoder_onnx = args.encoder_onnx or os.path.join(args.onnx_dir, "blip_encoder.onnx")
    decoder_onnx = args.decoder_onnx or os.path.join(args.onnx_dir, "blip_decoder.onnx")

    encoder_mlpackage = os.path.join(args.output_dir, "blip_encoder.mlpackage")
    decoder_mlpackage = os.path.join(args.output_dir, "blip_decoder.mlpackage")
    encoder_q4_mlpackage = os.path.join(args.output_dir, "blip_encoder_q4.mlpackage")
    decoder_q4_mlpackage = os.path.join(args.output_dir, "blip_decoder_q4.mlpackage")

    used_mode = args.conversion_mode
    if args.conversion_mode in {"auto", "onnx"}:
        try:
            convert_file(
                onnx_path=encoder_onnx,
                output_path=encoder_mlpackage,
                quantized_output_path=encoder_q4_mlpackage,
                compute_units=args.compute_units,
            )
            convert_file(
                onnx_path=decoder_onnx,
                output_path=decoder_mlpackage,
                quantized_output_path=decoder_q4_mlpackage,
                compute_units=args.compute_units,
            )
            used_mode = "onnx"
        except Exception as exc:
            if args.conversion_mode == "onnx":
                raise
            print(f"ONNX conversion unavailable ({exc}). Falling back to torch conversion.")
            used_mode = "torch"

    if used_mode == "torch":
        convert_from_torch(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            compute_units=args.compute_units,
            image_size=args.image_size,
            max_seq_len=args.max_seq_len,
        )

    print(f"Conversion mode used: {used_mode}")
    print(f"CoreML encoder: {encoder_mlpackage}")
    print(f"CoreML decoder: {decoder_mlpackage}")
    print(f"4-bit encoder: {encoder_q4_mlpackage}")
    print(f"4-bit decoder: {decoder_q4_mlpackage}")


if __name__ == "__main__":
    main()


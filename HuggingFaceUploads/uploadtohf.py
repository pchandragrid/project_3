from transformers import (
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    GitForCausalLM,
    GitProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)


def push_blip(
    local_dir: str = "saved_model_phase2",
    repo_id: str = "pchandragrid/blip-caption-model",
) -> None:
    model = BlipForConditionalGeneration.from_pretrained(local_dir)
    processor = BlipProcessor.from_pretrained(local_dir)
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)


def push_vit_gpt2(
    local_dir: str = "saved_vit_gpt2",
    repo_id: str = "pchandragrid/vit-gpt2-caption-model",
) -> None:
    model = VisionEncoderDecoderModel.from_pretrained(local_dir)
    image_processor = ViTImageProcessor.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model.push_to_hub(repo_id)
    image_processor.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)


def push_git(
    local_dir: str = "saved_git_model",
    repo_id: str = "pchandragrid/git-caption-model",
) -> None:
    model = GitForCausalLM.from_pretrained(local_dir)
    processor = GitProcessor.from_pretrained(local_dir)
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)


if __name__ == "__main__":
    push_blip()
    push_vit_gpt2()
    push_git()
    print("Uploaded: BLIP, ViT-GPT2, and GIT models.")
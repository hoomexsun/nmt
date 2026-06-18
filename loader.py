import torch
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)


# -------------------------------------------------------- MODEL LOADING
def add_custom_tokens(
    tokenizer, model, src_lang: str, tgt_lang: str, model_family: str
):
    """
    Add custom language tokens if missing and resize embeddings.
    """
    added = False

    for lang in [src_lang, tgt_lang]:
        if lang not in tokenizer.get_vocab():
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [lang]},
                replace_additional_special_tokens=False,
            )
            print(f"Added custom token: {lang}")
            added = True

    if added:
        if model_family == "mbart50":
            # Memory-efficient for mBART
            model.resize_token_embeddings(len(tokenizer))
            if hasattr(model, "final_logits_bias"):
                old_bias = model.final_logits_bias
                new_bias = torch.zeros(
                    1, len(tokenizer), dtype=old_bias.dtype, device=old_bias.device
                )
                new_bias[:, : old_bias.shape[1]] = old_bias
                model.final_logits_bias = new_bias
        else:
            # Full resize for NLLB
            model = manual_resize_token_embeddings(model, tokenizer)

        print(f"Resized embeddings to {len(tokenizer)} tokens")

    # Verify tokens
    for lang in [src_lang, tgt_lang]:
        lang_id = tokenizer.convert_tokens_to_ids(lang)
        if lang_id == tokenizer.unk_token_id:
            raise ValueError(f"Custom token {lang} was not added correctly.")

    return tokenizer, model


def manual_resize_token_embeddings(model, tokenizer):
    """Resize input/output embeddings manually after tokenizer expansion."""
    old_input_emb = model.get_input_embeddings()
    old_num_tokens, emb_dim = old_input_emb.weight.shape
    new_num_tokens = len(tokenizer)

    if new_num_tokens <= old_num_tokens:
        return model

    device = old_input_emb.weight.device
    dtype = old_input_emb.weight.dtype

    # 1. Resize input embeddings
    new_input_emb = nn.Embedding(
        new_num_tokens,
        emb_dim,
        padding_idx=old_input_emb.padding_idx,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_input_emb.weight[:old_num_tokens] = old_input_emb.weight
        new_input_emb.weight[old_num_tokens:].normal_(
            mean=0.0,
            std=getattr(model.config, "initializer_range", 0.02),
        )

    model.set_input_embeddings(new_input_emb)

    # 2. Resize output embeddings (lm_head)
    try:
        old_output_emb = model.get_output_embeddings()
        if old_output_emb is not None and hasattr(model, "set_output_embeddings"):
            out_vocab, out_dim = old_output_emb.weight.shape
            new_output_emb = nn.Linear(out_dim, new_num_tokens, bias=False).to(
                device=old_output_emb.weight.device,
                dtype=old_output_emb.weight.dtype,
            )
            with torch.no_grad():
                new_output_emb.weight[:out_vocab] = old_output_emb.weight
            model.set_output_embeddings(new_output_emb)
    except Exception:
        pass

    # 3. For mBART: resize final_logits_bias (fixes tensor size mismatch)
    if hasattr(model, "final_logits_bias"):
        old_bias = model.final_logits_bias
        new_bias = torch.zeros(1, new_num_tokens, dtype=dtype, device=device)
        new_bias[:, : old_bias.shape[1]] = old_bias
        model.final_logits_bias = new_bias

    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = new_num_tokens

    return model


def build_model_and_tokenizer(
    model_family: str, model_path: str, src_lang: str, tgt_lang: str
):
    if model_family == "mbart50":
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_path,
            src_lang="en_XX",
            tgt_lang="en_XX",
        )
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        print("\nSupported language codes (sample):")
        print(sorted(tokenizer.lang_code_to_id.keys())[:20])

    elif model_family == "nllb200-dist":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    return model, tokenizer


def load_for_training(cfg: dict):
    src_lang, tgt_lang = cfg["src_lang"], cfg["tgt_lang"]
    model, tokenizer = build_model_and_tokenizer(
        cfg["model_family"],
        cfg["hf_model_path"],
        src_lang,
        tgt_lang,
    )

    tokenizer, model = add_custom_tokens(
        tokenizer,
        model,
        src_lang,
        tgt_lang,
        cfg["model_family"],
    )

    # Verify token ids
    src_lang_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    print(f"\nSRC_LANG ({src_lang}) token id: {src_lang_id}")
    print(f"TGT_LANG ({tgt_lang}) token id: {tgt_lang_id}")

    model.config.forced_bos_token_id = tgt_lang_id

    return model, tokenizer


def load_for_inference(exp_dir: str, cfg: dict):
    src_lang, tgt_lang = cfg["src_lang"], cfg["tgt_lang"]
    model_family = cfg["model_family"]
    model, tokenizer = build_model_and_tokenizer(
        cfg["model_family"],
        exp_dir,
        src_lang,
        tgt_lang,
    )

    if model_family == "mbart50":
        for lang in [src_lang, tgt_lang]:
            if lang not in tokenizer.lang_code_to_id:
                lang_id = tokenizer.convert_tokens_to_ids(lang)
                if lang_id == tokenizer.unk_token_id:
                    raise ValueError(
                        f"Language token {lang} not found in tokenizer vocab for {exp_dir}"
                    )
                tokenizer.lang_code_to_id[lang] = lang_id
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    elif model_family == "nllb200-dist":
        if (
            hasattr(tokenizer, "lang_code_to_id")
            and tgt_lang in tokenizer.lang_code_to_id
        ):
            model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        else:
            model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer

"""Finding a text-to-image template's inputs by role, walking back from the sampler (what the
per-model workflows were made with, and what the importer will use). Plain API prompt dicts,
shaped like the converted Krea 2 / Z-Image templates; no ComfyUI needed."""

from zentrainer.apps.catalog import roles


def _krea_like():
    # prompt via a string primitive, an LLM enhancer behind a switch, size from a selector
    return {
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "x"}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 8, "cfg": 1, "model": ["unet", 0], "positive": ["pos", 0],
                                                    "negative": ["zero", 0], "latent_image": ["lat", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": ["show", 0], "clip": ["clip", 0]}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "show": {"class_type": "PreviewAny", "inputs": {"source": ["sw", 0]}},
        "sw": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["on", 0], "on_false": ["user", 0], "on_true": ["llm", 0]}},
        "on": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "llm": {"class_type": "TextGenerate", "inputs": {"prompt": ["cat", 0], "clip": ["clip", 0]}},
        "cat": {"class_type": "StringConcatenate", "inputs": {"string_a": ["system", 0], "string_b": ["user", 0]}},
        "system": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "You are a prompt engineer"}},
        "user": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "a hand holding the moon"}},
        "lat": {"class_type": "EmptyLatentImage", "inputs": {"width": ["res", 0], "height": ["res", 1], "batch_size": 1}},
        "res": {"class_type": "ResolutionSelector", "inputs": {"aspect_ratio": "1:1 (Square)", "megapixels": 1}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "k.safetensors"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "q.safetensors"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
    }


def test_roles_follow_strings_switches_and_selectors():
    r = roles(_krea_like())
    assert r["prompt"] == ("user", "value")  # the user's text, not the LLM's system prompt
    assert r["enhance"] == ("on", "value")
    assert (r["aspect_ratio"], r["megapixels"]) == (("res", "aspect_ratio"), ("res", "megapixels"))
    assert (r["seed"], r["steps"], r["cfg"]) == (("ks", "seed"), ("ks", "steps"), ("ks", "cfg"))
    assert "negative" not in r and "width" not in r  # zeroed-out negative; sized by the selector


def test_roles_plain_template():
    p = {
        "ks": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 4, "positive": ["p", 0], "negative": ["n", 0],
                                                    "latent_image": ["l", 0], "model": ["m", 0]}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["c", 0]}},
        "n": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["c", 0]}},
        "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 768, "batch_size": 1}},
        "m": {"class_type": "UNETLoader", "inputs": {"unet_name": "z.safetensors"}},
        "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t.safetensors"}},
    }
    r = roles(p)
    assert (r["prompt"], r["negative"], r["width"], r["height"]) == (("p", "text"), ("n", "text"), ("l", "width"), ("l", "height"))


def test_fixed_switches_fold_and_dead_branches_go():
    from zentrainer.apps.catalog import _fold_switches, _prune

    p = _krea_like()
    # Krea 2's template: a style LoRA behind a switch that's off
    p["style"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "krea2_darkbrush.safetensors"}}
    p["off"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": False}}
    p["msw"] = {"class_type": "ComfySwitchNode", "inputs": {"switch": ["off", 0], "on_false": ["unet", 0], "on_true": ["style", 0]}}
    p["ks"]["inputs"]["model"] = ["msw", 0]
    p["layout"] = {"class_type": "PreviewAny", "inputs": {"source": "unrelated"}}
    keep = {nid for nid, _ in roles(p).values()}
    _fold_switches(p, keep)
    _prune(p, ["save"])
    assert p["ks"]["inputs"]["model"] == ["unet", 0]
    assert not {"style", "off", "msw", "layout"} & set(p)
    assert {"sw", "on", "llm"} <= set(p)  # the prompt enhancer is a control: it stays

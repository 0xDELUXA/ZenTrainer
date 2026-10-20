"""Training adapter catalog: Hugging Face links, and local names."""

from zentrainer.models import adapters


def test_hf_links():
    m = adapters._HF_LINK.match
    assert m("https://huggingface.co/ostris/zimage_turbo_training_adapter").groups() == ("ostris/zimage_turbo_training_adapter", None)
    assert m("ostris/zimage_turbo_training_adapter").group(1) == "ostris/zimage_turbo_training_adapter"
    g = m("https://huggingface.co/ostris/x/blob/main/sub/a_v2.safetensors").groups()
    assert g == ("ostris/x", "sub/a_v2.safetensors")
    assert m("https://example.com/a/b") is None


def test_catalog_names_are_unique_files():
    names = [a.name for a in adapters.CURATED]
    assert len(names) == len(set(names))
    assert all(n.endswith(".safetensors") and "/" not in n for n in names)
    assert adapters.CURATED[0].suggest and "minimax-h3" in adapters.CURATED[0].families

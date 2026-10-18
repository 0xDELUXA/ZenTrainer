"""UI workflow -> API prompt conversion (zentrainer.apps.workflow). No ComfyUI needed: a tiny fake
object_info stands in for the server's, and the workflows are hand-made in the frontend's format."""

from zentrainer.apps.workflow import exposed_inputs, to_api, widget_values

OI = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
                               "output": ["MODEL", "CLIP", "VAE"]},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
                       "output": ["CONDITIONING"]},
    "KSampler": {"input": {"required": {
        "model": ["MODEL"], "seed": ["INT", {"default": 0, "control_after_generate": True}],
        "steps": ["INT", {"default": 20, "min": 1, "max": 100}], "cfg": ["FLOAT", {"default": 8.0}],
        "sampler_name": [["euler", "dpmpp_2m"]], "positive": ["CONDITIONING"], "latent_image": ["LATENT"]}},
        "output": ["LATENT"]},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {"default": 512}], "height": ["INT", {"default": 512}]}},
                         "output": ["LATENT"]},
    "LoadImage": {"input": {"required": {"image": [["x.png", "y.png"], {"image_upload": True}]}}, "output": ["IMAGE", "MASK"]},
    "ImageBlur": {"input": {"required": {"image": ["IMAGE"], "radius": ["INT", {"default": 1}]}}, "output": ["IMAGE"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
                  "output": [], "output_node": True},
    "StringNode": {"input": {"required": {"value": ["STRING"]}}, "output": ["STRING"]},
    "Resize": {"input": {"required": {"image": ["IMAGE"], "mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "scale", "inputs": {"required": {"factor": ["FLOAT", {"default": 2.0}]}}},
        {"key": "size", "inputs": {"required": {"width": ["INT"], "height": ["INT"]}}}]}]}}, "output": ["IMAGE"]},
}


def node(id, type, widgets=None, inputs=(), outputs=(), mode=0, **kw):
    return {"id": id, "type": type, "mode": mode, "widgets_values": widgets, "inputs": list(inputs),
            "outputs": list(outputs), **kw}


def inp(name, type, link=None, widget=False):
    return {"name": name, "type": type, "link": link, **({"widget": {"name": name}} if widget else {})}


def out(name, type, links=()):
    return {"name": name, "type": type, "links": list(links)}


def test_widget_mapping_skips_control_and_upload_slots():
    ks = {"widgets_values": [42, "randomize", 30, 6.5, "dpmpp_2m"]}
    assert widget_values(ks, OI["KSampler"]) == {"seed": 42, "steps": 30, "cfg": 6.5, "sampler_name": "dpmpp_2m"}
    assert widget_values({"widgets_values": ["y.png", "image"]}, OI["LoadImage"]) == {"image": "y.png"}
    # an old save without the control slot, a short list (defaults fill in), and a dict by name
    assert widget_values({"widgets_values": [7, 12]}, OI["KSampler"])["steps"] == 12
    assert widget_values({"widgets_values": [7]}, OI["KSampler"])["sampler_name"] == "euler"
    assert widget_values({"widgets_values": {"steps": 5}}, OI["KSampler"])["steps"] == 5


def test_dynamic_combo_flattens_to_dotted_keys():
    assert widget_values({"widgets_values": ["size", 640, 480]}, OI["Resize"]) == \
        {"mode": "size", "mode.width": 640, "mode.height": 480}


def _pipeline(blur_mode=0, extra_nodes=(), extra_links=(), save_link=3):
    """LoadImage 1 -> ImageBlur 2 -> SaveImage 3 (links 1, 2)."""
    nodes = [
        node(1, "LoadImage", ["x.png", "image"], outputs=[out("IMAGE", "IMAGE", [1])]),
        node(2, "ImageBlur", [3], [inp("image", "IMAGE", 1)], [out("IMAGE", "IMAGE", [2])], mode=blur_mode),
        node(3, "SaveImage", ["out"], [inp("images", "IMAGE", save_link)]),
        node(4, "Note", ["just a note"]),
        *extra_nodes,
    ]
    return {"nodes": nodes, "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"], *extra_links]}


def test_plain_graph_and_notes():
    api = to_api(_pipeline(save_link=2), OI)
    assert set(api) == {"1", "2", "3"}
    assert api["2"] == {"class_type": "ImageBlur", "inputs": {"radius": 3, "image": ["1", 0]},
                        "_meta": {"title": "ImageBlur"}}
    assert api["3"]["inputs"] == {"filename_prefix": "out", "images": ["2", 0]}


def test_bypass_routes_through_and_mute_drops():
    api = to_api(_pipeline(blur_mode=4, save_link=2), OI)
    assert "2" not in api and api["3"]["inputs"]["images"] == ["1", 0]
    api = to_api(_pipeline(blur_mode=2, save_link=2), OI)
    assert "2" not in api and "images" not in api["3"]["inputs"]


def test_reroute_is_followed():
    wf = _pipeline(extra_nodes=[node(5, "Reroute", None, [inp("", "*", 2)], [out("", "IMAGE", [3])])],
                   extra_links=[[3, 5, 0, 3, 0, "*"]])
    wf["links"][1] = [2, 2, 0, 5, 0, "IMAGE"]
    api = to_api(wf, OI)
    assert "5" not in api and api["3"]["inputs"]["images"] == ["2", 0]


def test_converted_widget_link_and_primitive_node():
    wf = {"nodes": [
        node(1, "CheckpointLoaderSimple", ["b.safetensors"], outputs=[out("MODEL", "MODEL", [1]), out("CLIP", "CLIP", [2])]),
        node(2, "StringNode", ["from a node"], outputs=[out("STRING", "STRING", [3])]),
        node(3, "CLIPTextEncode", ["typed text"], [inp("text", "STRING", 3, widget=True), inp("clip", "CLIP", 2)],
             [out("CONDITIONING", "CONDITIONING", [4])]),
        node(4, "EmptyLatentImage", [512, 512], outputs=[out("LATENT", "LATENT", [5])]),
        node(5, "PrimitiveNode", [1234, "fixed"], outputs=[{"name": "INT", "type": "INT", "links": [6], "widget": {"name": "seed"}}]),
        node(6, "KSampler", [1, "randomize", 20, 7.0, "euler"],
             [inp("model", "MODEL", 1), inp("positive", "CONDITIONING", 4), inp("latent_image", "LATENT", 5),
              inp("seed", "INT", 6, widget=True)]),
    ], "links": [[1, 1, 0, 6, 0, "MODEL"], [2, 1, 1, 3, 1, "CLIP"], [3, 2, 0, 3, 0, "STRING"],
                 [4, 3, 0, 6, 1, "CONDITIONING"], [5, 4, 0, 6, 2, "LATENT"], [6, 5, 0, 6, 3, "INT"]]}
    api = to_api(wf, OI)
    assert api["3"]["inputs"]["text"] == ["2", 0]  # the link wins over the widget value
    assert "5" not in api
    assert api["6"]["inputs"] == {"seed": 1234, "steps": 20, "cfg": 7.0, "sampler_name": "euler",
                                  "model": ["1", 0], "positive": ["3", 0], "latent_image": ["4", 0]}


def _subgraph_workflow(instance_values):
    """Root: Checkpoint 1 -> [instance 10 of "Encode"] -> KSampler 5.
    "Encode" holds CLIPTextEncode 1 (text promoted as "prompt") and an instance 2 of "Latent",
    which holds EmptyLatentImage 1 (width promoted as "width", then promoted again in "Encode")."""
    latent = {
        "id": "sg-latent", "name": "Latent",
        "inputs": [{"name": "width", "type": "INT", "linkIds": [1]}],
        "outputs": [{"name": "LATENT", "type": "LATENT", "linkIds": [2]}],
        "nodes": [node(1, "EmptyLatentImage", [1024, 576], [inp("width", "INT", 1, widget=True)],
                       [out("LATENT", "LATENT", [2])])],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0, "type": "INT"},
                  {"id": 2, "origin_id": 1, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "LATENT"}],
    }
    encode = {
        "id": "sg-encode", "name": "Encode",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [1], "label": "Prompt"},
                   {"name": "clip", "type": "CLIP", "linkIds": [2]},
                   {"name": "width", "type": "INT", "linkIds": [4]}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "linkIds": [3]},
                    {"name": "LATENT", "type": "LATENT", "linkIds": [5]}],
        "nodes": [node(1, "CLIPTextEncode", ["inner default"], [inp("text", "STRING", 1, widget=True), inp("clip", "CLIP", 2)],
                       [out("CONDITIONING", "CONDITIONING", [3])]),
                  node(2, "sg-latent", [], [inp("width", "INT", 4, widget=True)], [out("LATENT", "LATENT", [5])])],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0, "type": "STRING"},
                  {"id": 2, "origin_id": -10, "origin_slot": 1, "target_id": 1, "target_slot": 1, "type": "CLIP"},
                  {"id": 3, "origin_id": 1, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "CONDITIONING"},
                  {"id": 4, "origin_id": -10, "origin_slot": 2, "target_id": 2, "target_slot": 0, "type": "INT"},
                  {"id": 5, "origin_id": 2, "origin_slot": 0, "target_id": -20, "target_slot": 1, "type": "LATENT"}],
    }
    return {
        "id": "root-graph",
        "nodes": [
            node(1, "CheckpointLoaderSimple", ["a.safetensors"], outputs=[out("MODEL", "MODEL", [1]), out("CLIP", "CLIP", [2])]),
            # saved instance inputs are partial (only the linked one), so link 2's target_slot is 0
            node(10, "sg-encode", instance_values, [inp("clip", "CLIP", 2)],
                 [out("CONDITIONING", "CONDITIONING", [3]), out("LATENT", "LATENT", [4])], title="Encode it"),
            node(5, "KSampler", [7, "fixed", 10, 5.0, "euler"],
                 [inp("model", "MODEL", 1), inp("positive", "CONDITIONING", 3), inp("latent_image", "LATENT", 4)]),
        ],
        "links": [[1, 1, 0, 5, 0, "MODEL"], [2, 1, 1, 10, 0, "CLIP"], [3, 10, 0, 5, 1, "CONDITIONING"],
                  [4, 10, 1, 5, 2, "LATENT"]],
        "definitions": {"subgraphs": [encode, latent]},
        "extra": {"linearData": {
            "inputs": [["5", "steps"], ["10", "prompt"], ["1", "text"], ["root-graph:10:width", "width"],
                       ["99", "gone"]],
            "outputs": ["5", 1]}},
    }


def test_subgraphs_flatten_with_promoted_values():
    api = to_api(_subgraph_workflow(["a cat", 768]), OI)
    assert set(api) == {"1", "10:1", "10:2:1", "5"}
    assert api["10:1"]["inputs"] == {"text": "a cat", "clip": ["1", 1]}
    assert api["10:2:1"]["inputs"] == {"width": 768, "height": 576}  # promoted twice, set on the outer instance
    assert api["5"]["inputs"]["positive"] == ["10:1", 0] and api["5"]["inputs"]["latent_image"] == ["10:2:1", 0]
    # no values on the instance: the inner widgets keep theirs
    api = to_api(_subgraph_workflow([]), OI)
    assert api["10:1"]["inputs"]["text"] == "inner default" and api["10:2:1"]["inputs"]["width"] == 1024


def test_muted_subgraph_instance_drops_its_insides():
    wf = _subgraph_workflow([])
    wf["nodes"][1]["mode"] = 2
    api = to_api(wf, OI)
    assert set(api) == {"1", "5"} and "positive" not in api["5"]["inputs"]


def test_linear_data_resolves_to_api_ids():
    ex = exposed_inputs(_subgraph_workflow(["a cat", 768]), OI)
    by_ref = {str(i["ref"][0]): i for i in ex["inputs"]}
    steps = by_ref["5"]
    assert (steps["node_id"], steps["input"], steps["value"], steps["class_type"]) == ("5", "steps", 10, "KSampler")
    assert steps["spec"]["type"] == "INT" and steps["spec"]["max"] == 100
    prompt = by_ref["10"]
    assert (prompt["node_id"], prompt["input"], prompt["value"]) == ("10:1", "text", "a cat")
    assert prompt["label"] == "Prompt" and prompt["title"] == "Encode it" and prompt["spec"]["multiline"]
    # a legacy inner-node ref lands on the same promoted input
    assert (by_ref["1"]["node_id"], by_ref["1"]["input"]) == ("10:1", "text")
    width = by_ref["root-graph:10:width"]
    assert (width["node_id"], width["input"], width["value"]) == ("10:2:1", "width", 768)
    assert ex["unresolved"] == [["99", "gone"]]
    assert ex["outputs"] == [{"ref": "5", "node_ids": ["5"]}, {"ref": 1, "node_ids": ["1"]}]


def test_linear_data_on_a_primitive_points_at_its_target():
    wf = {"nodes": [
        node(1, "PrimitiveNode", [9, "fixed"], outputs=[{"name": "INT", "type": "INT", "links": [1], "widget": {"name": "radius"}}]),
        node(2, "Reroute", None, [inp("", "*", 1)], [out("", "INT", [2])]),
        node(3, "ImageBlur", [1], [inp("radius", "INT", 2, widget=True)]),
    ], "links": [[1, 1, 0, 2, 0, "INT"], [2, 2, 0, 3, 1, "INT"]],
        "extra": {"linearData": {"inputs": [["1", "value"]]}}}
    assert to_api(wf, OI) == {"3": {"class_type": "ImageBlur", "inputs": {"radius": 9}, "_meta": {"title": "ImageBlur"}}}
    (e,) = exposed_inputs(wf, OI)["inputs"]
    assert (e["node_id"], e["input"], e["value"]) == ("3", "radius", 9)

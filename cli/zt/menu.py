"""`zt` on its own (what the start scripts open): the logo, an update check, first-run setup, the menu."""

from __future__ import annotations

from pathlib import Path

from . import comfy, ports, repo, ui, update


def model_candidates() -> list[Path]:
    """Where an existing ComfyUI (or its models) usually lives: a fixed list, nothing searched."""
    home = Path.home()
    spots = [repo.ROOT.parent / "ComfyUI", home / "ComfyUI", home / "comfy" / "ComfyUI", home / "git" / "ComfyUI",
             home / "Documents" / "ComfyUI",  # ComfyUI Desktop
             home / "Desktop" / "ComfyUI_windows_portable" / "ComfyUI",
             home / "Downloads" / "ComfyUI_windows_portable" / "ComfyUI", Path("C:/ComfyUI"),
             Path("C:/ComfyUI_windows_portable/ComfyUI")]
    seen, out = set(), []
    for p in spots:
        try:
            if (p / "models").is_dir() and p.resolve() not in seen and p.resolve() != repo.COMFY.resolve():
                seen.add(p.resolve())
                out.append(p)
        except OSError:
            pass
    return out


def wizard() -> bool:
    """First run: where models are, which torch, which port; then setup. False if declined."""
    L = repo.lock()
    ui.say("First run: let's set up ZenTrainer's own ComfyUI. Your models are used where they are, never copied.")
    models: list[str] = []
    found = model_candidates()
    while True:
        choices = [(str(p), str(p)) for p in found if str(p.resolve()) not in models]
        choices += [("Another folder…", "other"), ("Done" if models else "Skip: no existing models", "done")]
        pick = ui.choose("Use models from an existing ComfyUI or models folder?", choices)
        if pick == "done":
            break
        path = Path(ui.folder("Folder:") if pick == "other" else pick).expanduser()
        if path.is_dir():
            models.append(str(path.resolve()))
            ui.ok(f"added {path}")
        else:
            ui.warn(f"{path} isn't a folder")
    for name, mem in comfy.gpus():
        ui.ok(f"found {name} ({mem / 2**30:.0f} GB)")
    detected = comfy.pick_torch(L["torch"]["variants"])
    labels = {"cpu": "CPU only (no GPU: very slow)"}
    torch = ui.choose("PyTorch build", [(f"{v}{'  (detected)' if v == detected else ''}" if v not in labels else labels[v], v)
                                        for v in L["torch"]["variants"]], detected)
    port = ui.ask("Port", str(repo.local().get("port") or L["port"]))
    if not port.isdigit() or not 0 < int(port) < 65536:
        ui.die(f"{port}: not a port number")
    ui.say(f"ComfyUI {L['comfyui']['describe']} and torch {L['torch']['version']}+{torch} go into {repo.COMFY} (a few GB).")
    if not ui.confirm("Go ahead?"):
        return False
    cfg = repo.local()
    cfg["port"] = int(port)
    repo.save_local(cfg)
    comfy.setup(models_from=models or None, torch=torch)
    return True


def settings() -> None:
    cfg = repo.local()
    cfg["port"] = int(ui.ask("Port", str(ports.configured())) or ports.configured())
    cfg["updates"] = ui.choose("Updates at start", [("Ask", "ask"), ("Always, without asking", "always"), ("Never check", "never")],
                               cfg.get("updates", "ask"))
    cfg["intro"] = ui.confirm("Play the intro?", cfg.get("intro", True))
    repo.save_local(cfg)
    ui.ok("saved")


def menu() -> None:
    ui.intro()
    update.offer()
    if not repo.venv_python().is_file() and not wizard():
        return
    while True:
        up = ports.running()
        status = f"running on :{up}" if up else "not running"
        ui.console.print(f"\n[dim]ZenTrainer is {status}[/]")
        pick = ui.choose("What now?", [
            ("Open ZenTrainer in the browser" if up else "Start ZenTrainer and open the browser", "open"),
            ("Monitor runs (live)", "monitor"),
            ("Start ZenTrainer here (this window shows its log)" if not up else "Restart ZenTrainer", "start"),
            ("Doctor: what's installed", "doctor"),
            ("Install extras (optimizers)", "extras"),
            ("Settings: port, updates, intro", "settings"),
            ("Quit", "quit"),
        ], "open")
        if pick == "open":  # started in the background, so the menu stays
            ui.open_browser(f"http://127.0.0.1:{up or comfy.background()}/zentrainer/")
        elif pick == "monitor":
            from . import monitor

            monitor.run(comfy.background())
        elif pick == "start":
            comfy.restart() if up else comfy.start()
        elif pick == "doctor":
            comfy.doctor()
        elif pick == "extras":
            groups = ui.checkboxes("Which extras? (space to tick)", list(comfy.EXTRAS))
            if groups:
                comfy.extras(groups)
        elif pick == "settings":
            settings()
        else:
            return

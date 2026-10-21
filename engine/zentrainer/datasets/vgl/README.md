# VGL prompts

Fibo 2.0's structured-caption schema and the prompts that write it, from Bria's
[VGL](https://github.com/Bria-AI/vgl) (commit 5eeecc2, dialect "thales"), under its MIT licence (LICENSE
here). ZenTrainer runs them on a local vision-language text encoder instead of VGL's API models.

- `schema.txt`: the keys a caption holds (`input_schema/generate.txt`)
- `caption_system.txt`, `caption_final.txt`: an image to a caption (`prompts/caption`)
- `generate_system.txt`, `generate_final.txt`: a short prompt to a caption (`prompts/generate`)

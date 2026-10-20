<script setup lang="ts">
// 3 Training: steps, optimizer, LR, rank, schedule and where the LoRAs go; under Advanced the
// optimizer's finer settings, the network, and the objective (with Focus for flow models).
import { ZenCombo, ZenField, ZenInput, ZenNumber, ZenSwitch, ZenTip } from '@nynxz/zenkit-ui'
import { computed, reactive } from 'vue'
import type { OptimizerInfo } from '@/api/engine'
import AdapterPicker from '@/components/AdapterPicker.vue'
import FocusField from '@/components/FocusField.vue'
import { fmtBytes } from '@/stores/downloads'
import { setSetting } from '@/stores/settings'
import { items, useRunFormContext } from './form'

const { spec, loraPlaces, choices, familyId, optimizers, dataset, opt, epochs, noise, resuming } = useRunFormContext()

const advanced = reactive({ training: false })
// ---- where checkpoints go: one of ComfyUI's loras folders ---------------------------------
const loraDir = computed({
  get: () => spec.output.lora_dir ?? loraPlaces.value.find((p) => p.default)?.path ?? '',
  set: (v: string) => {
    const place = loraPlaces.value.find((p) => p.path === v)
    spec.output.lora_dir = !place || place.default ? null : v
    setSetting('loraFolder', spec.output.lora_dir ?? '')
  },
})
const loraPlaceItems = computed(() =>
  loraPlaces.value.map((p) => ({ value: p.path, label: p.path, keywords: p.path, free: `${fmtBytes(p.free)} free` })),
)
const loraPlaceHint = computed(() => {
  const p = loraPlaces.value.find((x) => x.path === loraDir.value)
  if (!p) return 'One of ComfyUI’s LoRA folders'
  return p.default ? 'Inside this install. Your model library’s loras folder keeps them with the rest, visible to your other ComfyUI' : `In a zentrainer/ subfolder · ${fmtBytes(p.free)} free`
})
// ---- bound helpers ---------------------------------------------------------------------------
const alphaAuto = computed({
  get: () => spec.network.alpha == null,
  set: (v: boolean) => (spec.network.alpha = v ? null : spec.network.rank),
})
const alphaValue = computed({
  get: () => spec.network.alpha ?? spec.network.rank,
  set: (v: number) => (spec.network.alpha = v),
})
const initFrom = computed({
  get: () => spec.network.init_from ?? '',
  set: (v: string) => (spec.network.init_from = v || null),
})
const trainingAdapter = computed({
  get: () => spec.network.training_adapter ?? '',
  set: (v: string) => (spec.network.training_adapter = v || null),
})
const baseLora = computed({
  get: () => spec.network.base_lora ?? '',
  set: (v: string) => (spec.network.base_lora = v || null),
})
const targets = computed({
  get: () => (typeof spec.network.targets === 'string' ? spec.network.targets : 'attn_mlp'),
  set: (v: string) => (spec.network.targets = v as typeof spec.network.targets),
})
</script>

<template>
<section class="card sec">
  <h2><span class="num">3</span> Training</h2>
  <div class="grid-2">
    <ZenField label="Steps" stack :hint="epochs != null ? `≈ ${epochs.toFixed(1)} epochs of ${dataset?.steps_per_epoch} images` : ''">
      <ZenNumber v-model="spec.optim.steps" :min="1" :step="100" :precision="0" />
    </ZenField>
    <ZenField label="Optimizer" stack :tip="opt ? `${opt.note} Suggested LR ${opt.lr_hint}.`.trim() : ''">
      <ZenCombo
        v-model="spec.optim.optimizer"
        :items="optimizers.map((o) => ({ value: o.id, label: o.label, keywords: o.note, o }))"
        :sorted="false"
        :pick-guard="(v: string | number) => !optimizers.find((o) => o.id === v)?.available"
        :menu-width="360"
        :item-height="34"
      >
        <!-- what each one is for, while choosing: an ⓘ per option (clicking it doesn't pick) -->
        <template #option="{ item }">
          <div class="opt-row" :class="{ off: !(item.o as OptimizerInfo).available }">
            <span class="opt-name">{{ item.label }}</span>
            <span v-if="(item.o as OptimizerInfo).note" class="opt-tip" @click.stop @mousedown.stop @pointerdown.stop>
              <ZenTip :text="(item.o as OptimizerInfo).note" :label="`About ${item.label}`" />
            </span>
            <span class="spacer" />
            <span class="muted mono small">{{ (item.o as OptimizerInfo).available ? `lr ${(item.o as OptimizerInfo).lr_hint}` : 'not installed' }}</span>
          </div>
        </template>
      </ZenCombo>
    </ZenField>
    <ZenField label="Learning rate" stack>
      <ZenNumber v-model="spec.optim.learning_rate" :min="0" :step="0.00001" :precision="6" />
    </ZenField>
    <ZenField label="Rank" stack tip="8–16 for a character, 16–32 for a style">
      <ZenNumber v-model="spec.network.rank" :min="1" :max="256" :step="1" :precision="0" />
    </ZenField>
    <ZenField label="Schedule" stack>
      <ZenCombo v-model="spec.optim.scheduler" :items="items(['constant', 'linear', 'cosine'])" :searchable="false" :sorted="false" />
    </ZenField>
    <ZenField label="Save every" stack tip="0 = only the final LoRA">
      <ZenNumber v-model="spec.save_every" :min="0" :step="50" :precision="0" />
    </ZenField>
    <ZenField label="Save LoRAs to" stack class="span-2" :tip="loraPlaceHint">
      <ZenCombo v-model="loraDir" :items="loraPlaceItems" :sorted="false" :searchable="false" :item-height="40" :menu-width="520">
        <template #option="{ item }">
          <div class="place-opt">
            <span class="mono ellipsis">{{ item.label }}</span>
            <span class="spacer" />
            <span class="muted small mono">{{ item.free }}</span>
          </div>
        </template>
      </ZenCombo>
    </ZenField>
    <ZenField label="Resume states" stack tip="Optimizer state for resuming a stopped run; often bigger than the LoRA">
      <ZenCombo
        v-model="spec.output.keep_states"
        :items="[{ value: 'latest', label: 'Latest only' }, { value: 'all', label: 'Every checkpoint' }, { value: 'none', label: 'None (no resuming)' }]"
        :sorted="false"
        :searchable="false"
      />
    </ZenField>
  </div>
  <button class="more" @click="advanced.training = !advanced.training">
    <i :class="advanced.training ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'" /> Advanced training
  </button>
  <div v-if="advanced.training" class="adv">
    <div class="subgroup">
      <div class="sub-title">Optimizer</div>
      <div class="grid-2">
        <ZenField label="Warmup steps" stack><ZenNumber v-model="spec.optim.warmup_steps" :min="0" :step="10" :precision="0" /></ZenField>
        <ZenField label="Batch size" stack><ZenNumber v-model="spec.optim.batch_size" :min="1" :step="1" :precision="0" /></ZenField>
        <ZenField label="Grad accumulation" stack><ZenNumber v-model="spec.optim.grad_accumulation_steps" :min="1" :step="1" :precision="0" /></ZenField>
        <ZenField label="LoRA+" stack tip="The LoRA's up matrices learn this many times faster than the down ones: trains faster at the same stability. 4 is a safe start, the paper uses 16. Off for LoHa/LoKr/OFT">
          <div class="sw-row">
            <ZenSwitch :model-value="(spec.optim.lora_plus ?? 0) > 1" @update:model-value="spec.optim.lora_plus = $event ? 4 : 0" />
            <ZenNumber v-if="(spec.optim.lora_plus ?? 0) > 1" :model-value="spec.optim.lora_plus ?? 4" :min="1.5" :max="32" :step="1" :precision="1" @update:model-value="spec.optim.lora_plus = $event" />
            <span v-else class="sw-state">Off</span>
          </div>
        </ZenField>
        <ZenField label="Weight averaging" stack tip="Checkpoints, samples and the saved LoRA use a running average of the recent steps (EMA) instead of whichever step the save landed on: steadier likeness, less wobble between checkpoints. 0.98 ≈ the last 50 steps">
          <div class="sw-row">
            <ZenSwitch :model-value="(spec.optim.ema ?? 0) > 0" @update:model-value="spec.optim.ema = $event ? 0.98 : 0" />
            <ZenNumber v-if="(spec.optim.ema ?? 0) > 0" :model-value="spec.optim.ema ?? 0.98" :min="0.5" :max="0.999" :step="0.005" :precision="3" @update:model-value="spec.optim.ema = $event" />
            <span v-else class="sw-state">Off</span>
          </div>
        </ZenField>
        <ZenField v-if="spec.optim.optimizer === 'Automagic3'" label="LR floor (%)" stack tip="Automagic v3 never lowers its learning rate below this share of the start LR. Steps on fine detail (Focus toward Detail) give it noisy signals and it can talk itself down to almost nothing; 50% kept likeness climbing on MOSS. 0 = no floor">
          <ZenNumber :model-value="Math.round((spec.optim.lr_floor ?? 0) * 100)" :min="0" :max="100" :step="5" :precision="0" @update:model-value="spec.optim.lr_floor = $event / 100" />
        </ZenField>
        <ZenField label="Weight decay" stack><ZenNumber v-model="spec.optim.weight_decay" :min="0" :step="0.001" :precision="4" /></ZenField>
        <ZenField label="Grad clip" stack tip="Max grad norm; 0 = off"><ZenNumber v-model="spec.optim.max_grad_norm" :min="0" :step="0.1" :precision="2" /></ZenField>
      </div>
    </div>
    <div class="subgroup" :class="{ locked: resuming }">
      <div class="sub-title">Network</div>
      <div class="grid-2">
        <ZenField label="Algorithm" stack><ZenCombo v-model="spec.network.algorithm" :items="items(['LoRA', 'LoHa', 'LoKr', 'OFT'])" :searchable="false" :sorted="false" /></ZenField>
        <ZenField label="Layers" stack>
          <ZenCombo v-model="targets" :items="items(['attn', 'attn_mlp', 'all'], { attn: 'Attention', attn_mlp: 'Attention + MLP', all: 'Every layer' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Alpha" stack :tip="alphaAuto ? 'Auto = rank (scale 1.0)' : 'Scale is alpha / rank'">
          <div class="sw-row">
            <ZenSwitch v-model="alphaAuto" />
            <ZenNumber v-if="!alphaAuto" v-model="alphaValue" :min="0.01" :step="1" :precision="2" />
            <span v-else class="sw-state">Auto (= rank)</span>
          </div>
        </ZenField>
        <ZenField label="LoRA precision" stack tip="fp32: small steps land. bf16 rounds most AdamW steps away; use it only with Automagic (stochastic rounding)">
          <ZenCombo v-model="spec.network.lora_dtype" :items="items(['fp32', 'bf16'], { fp32: 'fp32 (recommended)', bf16: 'bf16' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Continue from" stack>
          <ZenCombo v-model="initFrom" :items="[{ value: '', label: 'New LoRA' }, ...items(choices.loras)]" :item-height="30" />
        </ZenField>
        <ZenField label="Training adapter" stack class="span-2" tip="A LoRA applied while training only, never in samples or the saved file. Guidance-distilled models (MiniMax H3) train much better on a de-distillation adapter">
          <AdapterPicker v-model="trainingAdapter" :family="familyId" />
        </ZenField>
        <ZenField label="Base LoRA" stack class="span-2" tip="A LoRA the new one sits on everywhere: training, samples and use. For a fix the model always needs (MOSS SFT's destripe) or an edit LoRA to build on. The new LoRA learns only what's left, so load the base beside it when you use it">
          <AdapterPicker v-model="baseLora" :family="familyId" kind="base" />
        </ZenField>
        <ZenField v-if="spec.network.base_lora" label="Base strength" stack tip="The strength the base LoRA runs at, in training and samples. Use it at the same strength">
          <ZenNumber :model-value="spec.network.base_strength ?? 1" :min="-4" :max="4" :step="0.05" :precision="2" @update:model-value="spec.network.base_strength = $event" />
        </ZenField>
        <ZenField label="Blocks" stack tip="Only these transformer blocks get adapters, e.g. 20-49: the backward stops at the first one, so steps are quicker, and the early blocks (layout, motion) stay untouched. Empty = every block">
          <ZenInput :model-value="spec.network.blocks ?? ''" placeholder="all" @update:model-value="spec.network.blocks = String($event).trim()" />
        </ZenField>
        <ZenField label="Train biases & norms" stack><div class="sw-row"><ZenSwitch v-model="spec.network.train_bias_norm" /><span class="sw-state">{{ spec.network.train_bias_norm ? 'On' : 'Off' }}</span></div></ZenField>
      </div>
    </div>
    <div class="subgroup">
      <div class="sub-title">Objective</div>
      <div class="grid-2">
        <ZenField label="Target" stack tip="Native: ε for SD, velocity for flow models">
          <ZenCombo v-model="spec.objective.target" :items="items(['native', 'x0'], { native: 'Model native', x0: 'Denoised (x0)' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField label="Timesteps" stack tip="Logit-normal favours mid noise (flow models)">
          <ZenCombo v-model="spec.objective.timesteps" :items="items(['uniform', 'logit_normal'], { logit_normal: 'Logit-normal' })" :searchable="false" :sorted="false" />
        </ZenField>
        <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Centre / spread" stack tip="Logit-normal's mean and standard deviation. Mean 0 centres on mid noise; positive leans to heavy noise (composition), negative to light noise (detail). MOSS pretrained at 1.09 / 0.95">
          <div class="inline">
            <ZenNumber v-model="spec.objective.logit_mean" :min="-3" :max="3" :step="0.1" :precision="2" />
            <ZenNumber v-model="spec.objective.logit_std" :min="0.1" :max="3" :step="0.1" :precision="2" />
          </div>
        </ZenField>
        <ZenField v-if="spec.objective.timesteps === 'logit_normal'" label="Uniform mix (%)" stack tip="Share of steps drawn uniformly instead: a little training at the extremes logit-normal rarely visits. 1% is plenty">
          <ZenNumber :model-value="Math.round((spec.objective.uniform_mix ?? 0) * 100)" :min="0" :max="50" :step="1" :precision="0" @update:model-value="spec.objective.uniform_mix = $event / 100" />
        </ZenField>
        <ZenField v-if="noise.known && noise.flow" label="Focus" stack class="full-row" tip="What the LoRA spends its steps on. Heavy noise teaches composition (layout, pose, colour), light noise teaches detail (faces, texture). Toward Detail usually helps likeness; the marker is the model's own schedule">
          <FocusField :objective="spec.objective" :profile="noise" />
        </ZenField>
        <ZenField v-else label="Timestep shift" stack tip="Flow models. 0 = the model's own (MOSS 5: composition, colour). 1–3 trains more on fine detail like faces">
          <ZenNumber v-model="spec.objective.shift" :min="0" :max="10" :step="0.5" :precision="1" />
        </ZenField>
        <ZenField label="Loss" stack><ZenCombo v-model="spec.objective.loss" :items="items(['mse', 'l1', 'huber', 'smooth_l1'])" :searchable="false" :sorted="false" /></ZenField>
        <ZenField label="Validation every" stack tip="Fixed-noise loss for a smooth curve; 0 = off">
          <ZenNumber v-model="spec.validation.every" :min="0" :step="50" :precision="0" />
        </ZenField>
      </div>
    </div>
  </div>
</section>
</template>

<style scoped src="./form.css"></style>
<style scoped>
.adv {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.subgroup {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.sub-title {
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted);
  padding-bottom: 6px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-border) 50%, transparent);
}
.place-opt {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
}
.place-opt .ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.opt-row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-width: 0;
  font-size: 12px;
}
.opt-tip {
  display: inline-flex;
}
.opt-row.off {
  opacity: 0.5;
}
</style>

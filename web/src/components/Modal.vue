<template>
  <div class="overlay" @click.self="$emit('close')">
    <div class="modal" :class="{ wide, xwide }">
      <div class="modal-head">
        <h3>{{ title }}</h3>
        <button class="ghost right" @click="$emit('close')">✕</button>
      </div>
      <slot />
      <div class="modal-foot" v-if="$slots.foot"><slot name="foot" /></div>
    </div>
  </div>
</template>

<script setup>
defineProps({ title: { type: String, default: '' }, wide: { type: Boolean, default: false }, xwide: { type: Boolean, default: false } })
defineEmits(['close'])
</script>

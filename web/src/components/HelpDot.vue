<template>
  <span class="help-dot" tabindex="0" :aria-label="text">
    ?
    <span class="help-tip">{{ text }}</span>
  </span>
</template>

<script setup>
defineProps({
  text: { type: String, default: '' },
})
</script>

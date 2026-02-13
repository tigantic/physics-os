<script>
  import { fly, fade } from 'svelte/transition';

  /** @type {string} key to trigger transition (typically $page.url.pathname) */
  export let key = '';
</script>

{#key key}
  <div class="page-transition"
    in:fly={{ y: 6, duration: 180, delay: 30 }}
    out:fade={{ duration: 80 }}>
    <slot />
  </div>
{/key}

<style>
  .page-transition {
    min-height: 100%;
  }
</style>

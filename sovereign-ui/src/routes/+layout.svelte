<script lang="ts">
	import '../app.css';
	import { websocket } from '$stores/websocket.svelte';
	import { processRegimeUpdate, loadInitialState } from '$stores/regime.svelte';
	import { updatePrimitives } from '$stores/primitives.svelte';
	import { addSignal } from '$stores/signals.svelte';
	import { onMount } from 'svelte';

	let { children } = $props();

	// Initialize WebSocket connection and fetch initial state on mount
	onMount(() => {
		console.log('[Layout] Mounting, fetching initial state...');
		
		// Fetch initial state via HTTP first (async IIFE)
		(async () => {
			try {
				const res = await fetch('/api/state');
				if (res.ok) {
					const data = await res.json();
					console.log('[Layout] Got initial state:', data.globalRegime);
					loadInitialState(data);
					
					// Also load primitives
					if (data.primitives) {
						updatePrimitives(data.primitives);
					}
				}
			} catch (e) {
				console.error('[Layout] Failed to fetch initial state:', e);
			}
		})();
		
		// Then connect WebSocket for updates
		console.log('[Layout] Connecting WebSocket...');
		websocket.connect();

		return () => {
			websocket.disconnect();
		};
	});

	// Process incoming messages - track messageCount to ensure reactivity
	$effect(() => {
		// Access messageCount to establish dependency
		const count = websocket.messageCount;
		const msg = websocket.lastMessage;
		if (!msg || count === 0) return;

		console.log('[Layout] Processing message #' + count + ':', msg.type);

		switch (msg.type) {
			case 'regime_update':
				console.log('[Layout] Calling processRegimeUpdate');
				processRegimeUpdate(msg);
				break;
			case 'primitive_update':
				console.log('[Layout] Calling updatePrimitives');
				updatePrimitives(msg.data.primitives);
				break;
			case 'signal':
				addSignal(msg.data);
				break;
			case 'heartbeat':
				// Heartbeat handled in websocket store
				break;
		}
	});
</script>

<div class="flex min-h-screen flex-col bg-void text-zinc-100">
	<!-- Connection status indicator -->
	{#if websocket.status !== 'connected'}
		<div
			class="fixed left-0 right-0 top-0 z-50 flex items-center justify-center gap-2 bg-zinc-900/90 px-4 py-2 text-sm backdrop-blur-sm"
		>
			{#if websocket.status === 'connecting'}
				<span class="h-2 w-2 animate-pulse rounded-full bg-amber-500"></span>
				<span class="text-amber-400">Connecting to Sovereign Daemon...</span>
			{:else if websocket.status === 'error'}
				<span class="h-2 w-2 rounded-full bg-red-500"></span>
				<span class="text-red-400">{websocket.error || 'Connection error'}</span>
				<button
					onclick={() => websocket.reconnect()}
					class="ml-2 rounded bg-red-500/20 px-2 py-0.5 text-red-400 hover:bg-red-500/30"
				>
					Retry
				</button>
			{:else}
				<span class="h-2 w-2 rounded-full bg-zinc-500"></span>
				<span class="text-zinc-400">Disconnected</span>
				<button
					onclick={() => websocket.connect()}
					class="ml-2 rounded bg-zinc-500/20 px-2 py-0.5 text-zinc-300 hover:bg-zinc-500/30"
				>
					Connect
				</button>
			{/if}
		</div>
	{/if}

	<!-- Main content -->
	<main class="flex-1">
		{@render children()}
	</main>

	<!-- Minimal footer with system status -->
	<footer class="border-t border-zinc-800/50 px-6 py-2">
		<div class="flex items-center justify-between text-xs text-zinc-500">
			<span class="font-mono">SOVEREIGN v2.1</span>
			<div class="flex items-center gap-4">
				<span
					class="flex items-center gap-1.5"
					class:text-green-500={websocket.isConnected}
					class:text-zinc-500={!websocket.isConnected}
				>
					<span
						class="h-1.5 w-1.5 rounded-full"
						class:bg-green-500={websocket.isConnected}
						class:bg-zinc-500={!websocket.isConnected}
					></span>
					{websocket.isConnected ? 'LIVE' : 'OFFLINE'}
				</span>
				<span class="font-mono">
					{new Date().toLocaleTimeString('en-US', { hour12: false })}
				</span>
			</div>
		</div>
	</footer>
</div>

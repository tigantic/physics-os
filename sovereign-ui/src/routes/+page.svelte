<script lang="ts">
	import RegimeHorizon from '$components/visualization/RegimeHorizon.svelte';
	import PrimitivesPanel from '$components/composite/PrimitivesPanel.svelte';
	import AssetsPanel from '$components/composite/AssetsPanel.svelte';
	import SignalsPanel from '$components/composite/SignalsPanel.svelte';
	import TensorManifold from '$components/visualization/TensorManifold.svelte';
	import { getRegimeLabel, getRegimeClass } from '$stores/regime.svelte';
	import { formatScore } from '$utils/format';
	import { onMount } from 'svelte';
	import { dataStore, startDataFetcher, stopDataFetcher } from '$stores/dataFetcher.svelte';
	
	// Server-loaded data (fallback)
	let { data } = $props();
	
	// Use live data if available, otherwise server data
	const regime = $derived(
		dataStore.updateCount > 0 
			? dataStore.globalRegime 
			: (data?.initialState?.globalRegime ?? 'UNKNOWN')
	);
	
	const confidence = $derived(
		dataStore.updateCount > 0 
			? dataStore.globalConfidence 
			: (data?.initialState?.globalConfidence ?? 0)
	);
	
	const assetCount = $derived(
		dataStore.updateCount > 0 
			? dataStore.assetList.length 
			: Object.keys(data?.initialState?.assets ?? {}).length
	);
	
	const assets = $derived(
		dataStore.updateCount > 0 
			? dataStore.assets 
			: (data?.initialState?.assets ?? {})
	);
	
	const primitives = $derived(
		dataStore.updateCount > 0 
			? dataStore.primitives 
			: (data?.initialState?.primitives ?? [])
	);
	
	// Connection status
	const modeLabel = $derived({
		'polling': '● HTTP',
		'websocket': '● WS',
		'disconnected': '○ ...'
	}[dataStore.mode] ?? '○ ...');
	
	const modeColor = $derived({
		'polling': 'text-amber-400',
		'websocket': 'text-green-400',
		'disconnected': 'text-zinc-500'
	}[dataStore.mode] ?? 'text-zinc-500');
	
	// Start data fetching on mount
	onMount(() => {
		startDataFetcher();
		return () => stopDataFetcher();
	});
</script>

<svelte:head>
	<title>Sovereign | {getRegimeLabel(regime)}</title>
</svelte:head>

<div class="flex h-screen flex-col overflow-hidden">
	<!-- Header: Regime Status -->
	<header class="flex items-center justify-between px-6 py-4">
		<div class="flex items-center gap-4">
			<h1 class="text-2xl font-light tracking-tight">
				<span class="text-zinc-500">Sovereign</span>
			</h1>
			<div
				class="flex items-center gap-2 rounded-full bg-surface/50 px-3 py-1 {getRegimeClass(regime)}"
			>
				<span
					class="h-2 w-2 rounded-full"
					class:bg-regime-stable={regime === 'MEAN_REVERTING'}
					class:bg-regime-trending={regime === 'TRENDING'}
					class:bg-regime-chaos={regime === 'CHAOTIC' || regime === 'CRASH'}
					class:bg-regime-transition={regime === 'TRANSITION'}
					class:bg-zinc-500={regime === 'UNKNOWN'}
				></span>
				<span class="text-sm font-medium">{getRegimeLabel(regime)}</span>
				<span class="text-xs text-zinc-500">{formatScore(confidence)}</span>
			</div>
		</div>

		<div class="flex items-center gap-4 text-sm text-zinc-500">
			<!-- Connection status indicator -->
			<span class="{modeColor} font-mono text-xs">
				{modeLabel}
				{#if dataStore.updateCount > 0}
					<span class="text-zinc-600">#{dataStore.updateCount}</span>
				{/if}
			</span>
			<span>
				{assetCount} assets
			</span>
		</div>
	</header>

	<!-- Main Grid -->
	<div class="grid flex-1 grid-cols-12 gap-4 overflow-hidden px-6 pb-4">
		<!-- Regime Horizon (full width) -->
		<section class="col-span-12 h-32 min-h-[100px]">
			<RegimeHorizon />
		</section>

		<!-- Left Column: Primitives -->
		<aside class="col-span-2 overflow-y-auto">
			<PrimitivesPanel initialPrimitives={primitives} />
		</aside>

		<!-- Center: Tensor Manifold -->
		<section class="col-span-6 min-h-[300px] overflow-hidden rounded-lg bg-surface/20">
			<TensorManifold />
		</section>

		<!-- Right Column: Assets + Signals -->
		<aside class="col-span-4 flex flex-col gap-4 overflow-hidden">
			<div class="flex-1 overflow-y-auto">
				<AssetsPanel initialAssets={assets} />
			</div>
			<div class="max-h-64 overflow-y-auto">
				<SignalsPanel />
			</div>
		</aside>
	</div>
</div>

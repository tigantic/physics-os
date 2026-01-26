/**
 * Server-side load function to fetch initial state
 * This runs on the server and passes data to the page
 */

export async function load({ fetch }) {
	try {
		const res = await fetch('http://localhost:8002/api/state');
		if (res.ok) {
			const state = await res.json();
			return {
				initialState: state
			};
		}
	} catch (e) {
		console.error('[+page.server] Failed to fetch state:', e);
	}

	return {
		initialState: null
	};
}

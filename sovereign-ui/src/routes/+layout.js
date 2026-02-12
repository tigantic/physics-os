// Disable SSR — this is a client-side SPA that talks to the backend API.
// SSR causes "css is not a function" in Svelte 5.50's SSR pipeline
// when running with runes:false (legacy mode).
export const ssr = false;

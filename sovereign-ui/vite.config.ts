import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://localhost:8002',
				changeOrigin: true,
				secure: false
			},
			'/ws': {
				target: 'http://localhost:8002',
				changeOrigin: true,
				ws: true,
				secure: false
			}
		}
	},
	optimizeDeps: {
		include: ['three', 'zod']
	}
});

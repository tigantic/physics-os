import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://127.0.0.1:8421',
				changeOrigin: true,
				secure: false,
				headers: {
					'X-API-Key': 'fp_QsU-wSv71x7KKxpNEjCxirFYtB76G7YrHNvq2C_nXgk'
				}
			},
			'/ws': {
				target: 'http://127.0.0.1:8421',
				changeOrigin: true,
				ws: true,
				secure: false
			}
		}
	},
	optimizeDeps: {
		include: ['three']
	}
});

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
	theme: {
		extend: {
			colors: {
				// Backgrounds
				void: '#08080A',
				surface: '#111114',
				elevated: '#1A1A1F',

				// Regimes
				regime: {
					stable: '#22C55E',
					trending: '#3B82F6',
					chaos: '#EF4444',
					transition: '#F59E0B'
				},

				// Primitives
				prim: {
					ot: '#8B5CF6',
					sgw: '#06B6D4',
					rmt: '#F97316',
					tg: '#EC4899',
					rkhs: '#84CC16',
					ph: '#14B8A6',
					ga: '#F43F5E'
				}
			},
			fontFamily: {
				sans: ['Inter Variable', 'SF Pro Display', 'system-ui', 'sans-serif'],
				mono: ['JetBrains Mono', 'SF Mono', 'monospace']
			},
			fontSize: {
				xs: 'clamp(0.625rem, 0.5rem + 0.25vw, 0.75rem)',
				sm: 'clamp(0.75rem, 0.65rem + 0.25vw, 0.875rem)',
				base: 'clamp(0.875rem, 0.8rem + 0.2vw, 1rem)',
				lg: 'clamp(1rem, 0.9rem + 0.3vw, 1.25rem)',
				xl: 'clamp(1.25rem, 1rem + 0.5vw, 1.75rem)',
				'2xl': 'clamp(1.5rem, 1.2rem + 0.75vw, 2.5rem)',
				'3xl': 'clamp(1.875rem, 1.5rem + 1vw, 3rem)'
			},
			transitionTimingFunction: {
				'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
				'in-out-sine': 'cubic-bezier(0.37, 0, 0.63, 1)'
			},
			transitionDuration: {
				fast: '100ms',
				normal: '200ms',
				slow: '400ms',
				glacial: '800ms'
			},
			animation: {
				'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
				'regime-shift': 'regime-shift 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
				'fade-in': 'fade-in 0.3s ease-out',
				'slide-up': 'slide-up 0.4s cubic-bezier(0.16, 1, 0.3, 1)'
			},
			keyframes: {
				'pulse-glow': {
					'0%, 100%': { opacity: '0.4' },
					'50%': { opacity: '0.8' }
				},
				'regime-shift': {
					'0%': { transform: 'scale(0.95)', opacity: '0.5' },
					'100%': { transform: 'scale(1)', opacity: '1' }
				},
				'fade-in': {
					'0%': { opacity: '0' },
					'100%': { opacity: '1' }
				},
				'slide-up': {
					'0%': { transform: 'translateY(10px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				}
			},
			boxShadow: {
				glow: '0 0 20px rgba(139, 92, 246, 0.15)',
				'glow-strong': '0 0 40px rgba(139, 92, 246, 0.25)'
			}
		}
	},
	plugins: []
};

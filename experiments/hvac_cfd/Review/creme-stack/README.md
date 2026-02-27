# Crème de la Crème Stack 🍰

The ultimate Next.js starter for enterprise-grade applications. Built with the best tools, zero compromises.

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript (strict mode) |
| Styling | Tailwind CSS |
| Components | shadcn/ui (Radix primitives) |
| Server State | TanStack Query v5 |
| Client State | Zustand |
| Forms | React Hook Form + Zod |
| API Client | openapi-fetch |
| Testing | Vitest + Testing Library + Playwright |
| Component Dev | Storybook |
| Icons | Lucide React |
| Dates | date-fns |
| Animation | Framer Motion |

## Getting Started

### Prerequisites

- Node.js 18.17+
- npm, yarn, or pnpm

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see your app.

## Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── (marketing)/        # Marketing pages (public)
│   ├── auth/               # Auth pages
│   ├── dashboard/          # Dashboard pages (protected)
│   ├── globals.css         # Global styles + design tokens
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Home page
├── components/
│   ├── ui/                 # shadcn/ui components
│   ├── common/             # Shared components
│   ├── layouts/            # Layout components
│   └── features/           # Feature-specific components
├── hooks/                  # Custom React hooks
├── lib/
│   ├── api/                # API client & utilities
│   ├── providers/          # React context providers
│   └── utils.ts            # Utility functions
├── stores/                 # Zustand stores
├── types/                  # TypeScript type definitions
└── test/                   # Test utilities
```

## Available Scripts

```bash
# Development
npm run dev           # Start dev server
npm run build         # Build for production
npm run start         # Start production server
npm run lint          # Run ESLint
npm run lint:fix      # Fix ESLint errors
npm run format        # Format with Prettier
npm run type-check    # TypeScript check

# Testing
npm run test          # Run unit tests
npm run test:ui       # Run tests with UI
npm run test:coverage # Run tests with coverage
npm run test:e2e      # Run E2E tests
npm run test:e2e:ui   # Run E2E tests with UI

# Storybook
npm run storybook     # Start Storybook
npm run build-storybook # Build Storybook

# API
npm run generate-api  # Generate API types from OpenAPI spec
```

## Adding Components

This project uses [shadcn/ui](https://ui.shadcn.com). To add a component:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add dialog
# etc.
```

## API Integration

1. Place your OpenAPI spec at `./openapi.yaml`
2. Run `npm run generate-api` to generate types
3. Import and use the typed client:

```typescript
import { api } from '@/lib/api/client';

// Fully typed!
const { data, error } = await api.GET('/users/{id}', {
  params: { path: { id: '123' } }
});
```

## State Management

### Server State (API data) - TanStack Query

```typescript
import { useQuery, useMutation } from '@tanstack/react-query';

function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => api.GET('/users'),
  });
}
```

### Client State (UI state) - Zustand

```typescript
import { useAuthStore } from '@/stores';

function Component() {
  const { user, logout } = useAuthStore();
  // ...
}
```

## Forms

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

function LoginForm() {
  const form = useForm({
    resolver: zodResolver(schema),
  });
  // ...
}
```

## Customization

### Design Tokens

Edit `src/app/globals.css` to customize:

- Colors (light and dark mode)
- Border radius
- Other design tokens

### Tailwind

Edit `tailwind.config.ts` to customize:

- Fonts
- Spacing
- Animations
- etc.

## Deployment

### Vercel (Recommended)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/your-repo)

### Other Platforms

```bash
npm run build
npm run start
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with 💜 using the Crème de la Crème Stack

import Link from 'next/link';
import { Button } from '@/components/ui';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui';
import { ArrowRight, Zap, Shield, Sparkles, Code2 } from 'lucide-react';

export default function HomePage() {
  return (
    <main id="main-content" className="min-h-screen">
      {/* Hero Section */}
      <section className="container flex flex-col items-center justify-center gap-6 py-24 text-center md:py-32">
        <div className="flex items-center gap-2 rounded-full border bg-muted px-4 py-1.5 text-sm font-medium">
          <Sparkles className="h-4 w-4" />
          <span>The Crème de la Crème Stack</span>
        </div>

        <h1 className="max-w-4xl text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
          Build{' '}
          <span className="bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
            Enterprise-Grade
          </span>{' '}
          Apps at Lightning Speed
        </h1>

        <p className="max-w-2xl text-lg text-muted-foreground md:text-xl">
          Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query,
          Zustand, React Hook Form + Zod. Everything you need, nothing you don&apos;t.
        </p>

        <div className="flex flex-col gap-4 sm:flex-row">
          <Button size="lg" asChild>
            <Link href="/dashboard">
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button variant="outline" size="lg" asChild>
            <Link href="https://github.com/your-repo" target="_blank">
              <Code2 className="mr-2 h-4 w-4" />
              View on GitHub
            </Link>
          </Button>
        </div>
      </section>

      {/* Features Section */}
      <section className="container py-16">
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <FeatureCard
            icon={<Zap className="h-6 w-6" />}
            title="Blazing Fast"
            description="Built on Next.js 14 with React Server Components for optimal performance and SEO."
          />
          <FeatureCard
            icon={<Shield className="h-6 w-6" />}
            title="Type Safe"
            description="End-to-end TypeScript with strict mode. Catch bugs before they reach production."
          />
          <FeatureCard
            icon={<Sparkles className="h-6 w-6" />}
            title="Beautiful UI"
            description="shadcn/ui components with Tailwind CSS. Dark mode included out of the box."
          />
        </div>
      </section>

      {/* Stack Section */}
      <section className="container py-16">
        <h2 className="mb-8 text-center text-3xl font-bold">The Stack</h2>
        <div className="mx-auto grid max-w-4xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            'Next.js 14',
            'TypeScript',
            'Tailwind CSS',
            'shadcn/ui',
            'TanStack Query',
            'Zustand',
            'React Hook Form',
            'Zod',
          ].map((tech) => (
            <div
              key={tech}
              className="flex items-center justify-center rounded-lg border bg-card p-4 text-center font-medium"
            >
              {tech}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

// ============================================
// FEATURE CARD COMPONENT
// ============================================

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
          {icon}
        </div>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
    </Card>
  );
}

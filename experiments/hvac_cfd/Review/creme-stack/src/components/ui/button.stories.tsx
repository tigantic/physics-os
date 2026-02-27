import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './button';
import { Mail, ArrowRight, Loader2 } from 'lucide-react';

const meta: Meta<typeof Button> = {
  title: 'UI/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: [
        'default',
        'destructive',
        'outline',
        'secondary',
        'ghost',
        'link',
        'success',
        'warning',
      ],
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg', 'xl', 'icon', 'icon-sm', 'icon-lg'],
    },
    disabled: {
      control: 'boolean',
    },
    loading: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

// ============================================
// STORIES
// ============================================

export const Default: Story = {
  args: {
    children: 'Button',
  },
};

export const Variants: Story = {
  render: () => (
    <div className="flex flex-wrap gap-4">
      <Button variant="default">Default</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="outline">Outline</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="link">Link</Button>
      <Button variant="destructive">Destructive</Button>
      <Button variant="success">Success</Button>
      <Button variant="warning">Warning</Button>
    </div>
  ),
};

export const Sizes: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-4">
      <Button size="sm">Small</Button>
      <Button size="default">Default</Button>
      <Button size="lg">Large</Button>
      <Button size="xl">Extra Large</Button>
    </div>
  ),
};

export const IconButtons: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-4">
      <Button size="icon-sm">
        <Mail className="h-4 w-4" />
      </Button>
      <Button size="icon">
        <Mail className="h-4 w-4" />
      </Button>
      <Button size="icon-lg">
        <Mail className="h-5 w-5" />
      </Button>
    </div>
  ),
};

export const WithIcons: Story = {
  render: () => (
    <div className="flex flex-wrap gap-4">
      <Button leftIcon={<Mail className="h-4 w-4" />}>With Left Icon</Button>
      <Button rightIcon={<ArrowRight className="h-4 w-4" />}>
        With Right Icon
      </Button>
      <Button
        leftIcon={<Mail className="h-4 w-4" />}
        rightIcon={<ArrowRight className="h-4 w-4" />}
      >
        Both Icons
      </Button>
    </div>
  ),
};

export const Loading: Story = {
  args: {
    children: 'Loading...',
    loading: true,
  },
};

export const Disabled: Story = {
  args: {
    children: 'Disabled',
    disabled: true,
  },
};

export const AsChild: Story = {
  render: () => (
    <Button asChild>
      <a href="https://example.com">Link styled as button</a>
    </Button>
  ),
};

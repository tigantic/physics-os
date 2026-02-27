/**
 * DashboardShell Component Tests
 * 
 * Tests layout components: PageHeader, Breadcrumb, SectionCard, etc.
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '@/test/utils';
import {
  DashboardShell,
  PageHeader,
  Breadcrumb,
  SectionCard,
  StatCard,
  EmptyState,
} from './index';

describe('Breadcrumb', () => {
  it('should render breadcrumb items', () => {
    const items = [
      { label: 'Home', href: '/' },
      { label: 'Simulations', href: '/simulations' },
      { label: 'Detail' },
    ];

    renderWithProviders(<Breadcrumb items={items} />);

    expect(screen.getByText('Home')).toBeInTheDocument();
    expect(screen.getByText('Simulations')).toBeInTheDocument();
    expect(screen.getByText('Detail')).toBeInTheDocument();
  });

  it('should render links for items with href', () => {
    const items = [
      { label: 'Home', href: '/' },
      { label: 'Current' },
    ];

    renderWithProviders(<Breadcrumb items={items} />);

    const homeLink = screen.getByText('Home');
    expect(homeLink.closest('a')).toHaveAttribute('href', '/');
  });

  it('should render plain text for items without href', () => {
    const items = [{ label: 'Current Page' }];

    renderWithProviders(<Breadcrumb items={items} />);

    const current = screen.getByText('Current Page');
    expect(current.tagName).toBe('SPAN');
  });
});

describe('PageHeader', () => {
  it('should render title', () => {
    renderWithProviders(<PageHeader title="Test Page" />);
    
    expect(screen.getByRole('heading', { name: 'Test Page' })).toBeInTheDocument();
  });

  it('should render description when provided', () => {
    renderWithProviders(
      <PageHeader title="Test" description="This is a description" />
    );
    
    expect(screen.getByText('This is a description')).toBeInTheDocument();
  });

  it('should render actions when provided', () => {
    renderWithProviders(
      <PageHeader
        title="Test"
        actions={<button>Action Button</button>}
      />
    );
    
    expect(screen.getByRole('button', { name: 'Action Button' })).toBeInTheDocument();
  });

  it('should render breadcrumbs when provided', () => {
    renderWithProviders(
      <PageHeader
        title="Detail"
        breadcrumbs={[
          { label: 'Home', href: '/' },
          { label: 'Detail' },
        ]}
      />
    );
    
    expect(screen.getByText('Home')).toBeInTheDocument();
  });
});

describe('DashboardShell', () => {
  it('should render children', () => {
    renderWithProviders(
      <DashboardShell>
        <div data-testid="child-content">Content</div>
      </DashboardShell>
    );
    
    expect(screen.getByTestId('child-content')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    renderWithProviders(
      <DashboardShell className="custom-class">
        <div>Content</div>
      </DashboardShell>
    );
    
    const main = screen.getByRole('main');
    expect(main).toHaveClass('custom-class');
  });
});

describe('SectionCard', () => {
  it('should render with title', () => {
    renderWithProviders(<SectionCard title="Section Title">Content</SectionCard>);
    
    expect(screen.getByText('Section Title')).toBeInTheDocument();
  });

  it('should render description when provided', () => {
    renderWithProviders(
      <SectionCard title="Title" description="Section description">
        Content
      </SectionCard>
    );
    
    expect(screen.getByText('Section description')).toBeInTheDocument();
  });

  it('should render actions when provided', () => {
    renderWithProviders(
      <SectionCard title="Title" actions={<button>Add</button>}>
        Content
      </SectionCard>
    );
    
    expect(screen.getByRole('button', { name: 'Add' })).toBeInTheDocument();
  });

  it('should render children content', () => {
    renderWithProviders(
      <SectionCard title="Title">
        <p data-testid="section-content">Section content here</p>
      </SectionCard>
    );
    
    expect(screen.getByTestId('section-content')).toBeInTheDocument();
  });
});

describe('StatCard', () => {
  it('should render title and value', () => {
    renderWithProviders(
      <StatCard title="Active Simulations" value={5} />
    );
    
    expect(screen.getByText('Active Simulations')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('should render description when provided', () => {
    renderWithProviders(
      <StatCard title="Title" value={10} description="Additional info" />
    );
    
    expect(screen.getByText('Additional info')).toBeInTheDocument();
  });

  it('should render icon when provided', () => {
    const Icon = () => <span data-testid="stat-icon">📊</span>;
    renderWithProviders(
      <StatCard title="Stats" value={42} icon={<Icon />} />
    );
    
    expect(screen.getByTestId('stat-icon')).toBeInTheDocument();
  });

  it('should render positive trend', () => {
    renderWithProviders(
      <StatCard
        title="Revenue"
        value="$1000"
        trend={{ value: 15, isPositive: true }}
      />
    );
    
    expect(screen.getByText(/15%/)).toBeInTheDocument();
  });

  it('should render negative trend', () => {
    renderWithProviders(
      <StatCard
        title="Errors"
        value={3}
        trend={{ value: 10, isPositive: false }}
      />
    );
    
    expect(screen.getByText(/10%/)).toBeInTheDocument();
  });
});

describe('EmptyState', () => {
  it('should render title and description', () => {
    renderWithProviders(
      <EmptyState
        title="No simulations"
        description="Create your first simulation to get started"
      />
    );
    
    expect(screen.getByText('No simulations')).toBeInTheDocument();
    expect(screen.getByText('Create your first simulation to get started')).toBeInTheDocument();
  });

  it('should render icon when provided', () => {
    const Icon = () => <span data-testid="empty-icon">📭</span>;
    renderWithProviders(
      <EmptyState
        icon={<Icon />}
        title="Empty"
        description="Nothing here"
      />
    );
    
    expect(screen.getByTestId('empty-icon')).toBeInTheDocument();
  });

  it('should render action when provided', () => {
    renderWithProviders(
      <EmptyState
        title="No data"
        description="Add some data"
        action={<button>Add Data</button>}
      />
    );
    
    expect(screen.getByRole('button', { name: 'Add Data' })).toBeInTheDocument();
  });
});

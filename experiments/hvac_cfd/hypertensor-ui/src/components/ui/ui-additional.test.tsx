/**
 * Additional UI Components Tests
 * 
 * Tests for Checkbox, Label, Switch, Skeleton, Separator
 * Constitutional Compliance: Article III Testing Protocols
 */

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Checkbox } from './checkbox';
import { Label } from './label';
import { Switch } from './switch';
import { Skeleton } from './skeleton';
import { Separator } from './separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './table';

// ============================================
// CHECKBOX TESTS
// ============================================

describe('Checkbox', () => {
  it('should render unchecked by default', () => {
    render(<Checkbox />);
    
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).not.toBeChecked();
  });

  it('should be checkable', () => {
    const onChange = vi.fn();
    render(<Checkbox onCheckedChange={onChange} />);
    
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);
    
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it('should render checked state', () => {
    render(<Checkbox checked />);
    
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toHaveAttribute('data-state', 'checked');
  });

  it('should be disabled when disabled prop is true', () => {
    render(<Checkbox disabled />);
    
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeDisabled();
  });

  it('should apply custom className', () => {
    render(<Checkbox className="custom-checkbox" />);
    
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toHaveClass('custom-checkbox');
  });
});

// ============================================
// LABEL TESTS
// ============================================

describe('Label', () => {
  it('should render with text content', () => {
    render(<Label>Email Address</Label>);
    
    expect(screen.getByText('Email Address')).toBeInTheDocument();
  });

  it('should render required indicator', () => {
    render(<Label required>Username</Label>);
    
    expect(screen.getByText('*')).toBeInTheDocument();
  });

  it('should render optional indicator', () => {
    render(<Label optional>Nickname</Label>);
    
    expect(screen.getByText('(optional)')).toBeInTheDocument();
  });

  it('should support htmlFor prop', () => {
    render(<Label htmlFor="email-input">Email</Label>);
    
    const label = screen.getByText('Email');
    expect(label).toHaveAttribute('for', 'email-input');
  });

  it('should apply custom className', () => {
    render(<Label className="custom-label">Test</Label>);
    
    expect(screen.getByText('Test')).toHaveClass('custom-label');
  });
});

// ============================================
// SWITCH TESTS
// ============================================

describe('Switch', () => {
  it('should render unchecked by default', () => {
    render(<Switch />);
    
    const switchEl = screen.getByRole('switch');
    expect(switchEl).toBeInTheDocument();
    expect(switchEl).toHaveAttribute('data-state', 'unchecked');
  });

  it('should toggle on click', () => {
    const onCheckedChange = vi.fn();
    render(<Switch onCheckedChange={onCheckedChange} />);
    
    const switchEl = screen.getByRole('switch');
    fireEvent.click(switchEl);
    
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it('should render checked state', () => {
    render(<Switch checked />);
    
    const switchEl = screen.getByRole('switch');
    expect(switchEl).toHaveAttribute('data-state', 'checked');
  });

  it('should be disabled when disabled prop is true', () => {
    render(<Switch disabled />);
    
    const switchEl = screen.getByRole('switch');
    expect(switchEl).toBeDisabled();
  });

  it('should apply custom className', () => {
    render(<Switch className="custom-switch" />);
    
    const switchEl = screen.getByRole('switch');
    expect(switchEl).toHaveClass('custom-switch');
  });
});

// ============================================
// SKELETON TESTS
// ============================================

describe('Skeleton', () => {
  it('should render a div element', () => {
    const { container } = render(<Skeleton />);
    
    const skeleton = container.firstChild;
    expect(skeleton).toBeInstanceOf(HTMLDivElement);
  });

  it('should have animation class', () => {
    const { container } = render(<Skeleton />);
    
    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton).toHaveClass('animate-pulse');
  });

  it('should apply custom className', () => {
    const { container } = render(<Skeleton className="h-12 w-48" />);
    
    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton).toHaveClass('h-12', 'w-48');
  });

  it('should spread additional props', () => {
    const { container } = render(<Skeleton data-testid="skeleton-test" />);
    
    const skeleton = container.firstChild as HTMLElement;
    expect(skeleton).toHaveAttribute('data-testid', 'skeleton-test');
  });
});

// ============================================
// SEPARATOR TESTS
// ============================================

describe('Separator', () => {
  it('should render horizontal by default', () => {
    const { container } = render(<Separator />);
    
    const separator = container.firstChild as HTMLElement;
    expect(separator).toBeInTheDocument();
    expect(separator).toHaveAttribute('data-orientation', 'horizontal');
  });

  it('should render vertical orientation', () => {
    const { container } = render(<Separator orientation="vertical" />);
    
    const separator = container.firstChild as HTMLElement;
    expect(separator).toHaveAttribute('data-orientation', 'vertical');
  });

  it('should apply custom className', () => {
    const { container } = render(<Separator className="my-custom-separator" />);
    
    const separator = container.firstChild as HTMLElement;
    expect(separator).toHaveClass('my-custom-separator');
  });

  it('should have decorative role by default', () => {
    const { container } = render(<Separator />);
    
    const separator = container.firstChild as HTMLElement;
    // Radix separator is decorative by default (role="none")
    expect(separator).toHaveAttribute('role', 'none');
  });
});

// ============================================
// SLIDER TESTS - Skipped due to ResizeObserver dependency
// These are tested via E2E tests instead
// ============================================

// Slider requires ResizeObserver which isn't available in jsdom

// ============================================
// TABLE TESTS
// ============================================

describe('Table', () => {
  it('should render a table', () => {
    render(
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Header</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell>Cell</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
    
    expect(screen.getByRole('table')).toBeInTheDocument();
  });

  it('should render table headers', () => {
    render(
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Email</TableHead>
          </TableRow>
        </TableHeader>
      </Table>
    );
    
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Email')).toBeInTheDocument();
  });

  it('should render table rows and cells', () => {
    render(
      <Table>
        <TableBody>
          <TableRow>
            <TableCell>John</TableCell>
            <TableCell>john@example.com</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
    
    expect(screen.getByText('John')).toBeInTheDocument();
    expect(screen.getByText('john@example.com')).toBeInTheDocument();
  });

  it('should apply custom className to Table', () => {
    const { container } = render(
      <Table className="custom-table">
        <TableBody>
          <TableRow>
            <TableCell>Test</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
    
    expect(container.querySelector('.custom-table')).toBeInTheDocument();
  });

  it('should render multiple rows', () => {
    render(
      <Table>
        <TableBody>
          <TableRow>
            <TableCell>Row 1</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Row 2</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Row 3</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
    
    expect(screen.getByText('Row 1')).toBeInTheDocument();
    expect(screen.getByText('Row 2')).toBeInTheDocument();
    expect(screen.getByText('Row 3')).toBeInTheDocument();
  });
});

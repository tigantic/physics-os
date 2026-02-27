/**
 * UI Components Tests
 * 
 * Tests for Button, Card, Dialog, Input, Select, Tabs, DropdownMenu
 * Constitutional Compliance: Article III Testing Protocols
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// ============================================
// BUTTON COMPONENT TESTS
// ============================================

describe('Button', () => {
  it('should render button with text', async () => {
    const { Button } = await import('./button');
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
  });

  it('should handle click events', async () => {
    const { Button } = await import('./button');
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click</Button>);
    
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('should render disabled state', async () => {
    const { Button } = await import('./button');
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('should render different variants', async () => {
    const { Button } = await import('./button');
    const { rerender } = render(<Button variant="default">Default</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-primary');
    
    rerender(<Button variant="destructive">Destructive</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-destructive');
    
    rerender(<Button variant="outline">Outline</Button>);
    expect(screen.getByRole('button')).toHaveClass('border');
    
    rerender(<Button variant="ghost">Ghost</Button>);
    expect(screen.getByRole('button')).toHaveClass('hover:bg-accent');
  });

  it('should render different sizes', async () => {
    const { Button } = await import('./button');
    const { rerender } = render(<Button size="sm">Small</Button>);
    expect(screen.getByRole('button')).toHaveClass('h-9');
    
    rerender(<Button size="lg">Large</Button>);
    expect(screen.getByRole('button')).toHaveClass('h-11');
  });

  it('should support as child pattern', async () => {
    const { Button } = await import('./button');
    render(
      <Button asChild>
        <a href="/test">Link Button</a>
      </Button>
    );
    expect(screen.getByRole('link', { name: /link button/i })).toBeInTheDocument();
  });
});

// ============================================
// INPUT COMPONENT TESTS
// ============================================

describe('Input', () => {
  it('should render input', async () => {
    const { Input } = await import('./input');
    render(<Input placeholder="Enter text" />);
    expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('should accept user input', async () => {
    const { Input } = await import('./input');
    const handleChange = vi.fn();
    render(<Input onChange={handleChange} />);
    
    const input = screen.getByRole('textbox');
    await userEvent.type(input, 'test');
    
    expect(handleChange).toHaveBeenCalled();
  });

  it('should render disabled state', async () => {
    const { Input } = await import('./input');
    render(<Input disabled />);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });

  it('should support different types', async () => {
    const { Input } = await import('./input');
    const { rerender } = render(<Input type="email" />);
    expect(screen.getByRole('textbox')).toHaveAttribute('type', 'email');
    
    rerender(<Input type="password" />);
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
  });
});

// ============================================
// CARD COMPONENT TESTS
// ============================================

describe('Card', () => {
  it('should render card', async () => {
    const { Card, CardHeader, CardTitle, CardContent } = await import('./card');
    render(
      <Card>
        <CardHeader>
          <CardTitle>Test Card</CardTitle>
        </CardHeader>
        <CardContent>Content</CardContent>
      </Card>
    );
    
    expect(screen.getByText('Test Card')).toBeInTheDocument();
    expect(screen.getByText('Content')).toBeInTheDocument();
  });

  it('should render card description', async () => {
    const { Card, CardHeader, CardTitle, CardDescription } = await import('./card');
    render(
      <Card>
        <CardHeader>
          <CardTitle>Title</CardTitle>
          <CardDescription>Description text</CardDescription>
        </CardHeader>
      </Card>
    );
    
    expect(screen.getByText('Description text')).toBeInTheDocument();
  });

  it('should render card footer', async () => {
    const { Card, CardFooter } = await import('./card');
    render(
      <Card>
        <CardFooter>Footer content</CardFooter>
      </Card>
    );
    
    expect(screen.getByText('Footer content')).toBeInTheDocument();
  });
});

// ============================================
// DIALOG COMPONENT TESTS
// ============================================

describe('Dialog', () => {
  it('should open on trigger click', async () => {
    const { Dialog, DialogTrigger, DialogContent, DialogTitle } = await import('./dialog');
    render(
      <Dialog>
        <DialogTrigger>Open Dialog</DialogTrigger>
        <DialogContent>
          <DialogTitle>Dialog Title</DialogTitle>
        </DialogContent>
      </Dialog>
    );
    
    fireEvent.click(screen.getByText('Open Dialog'));
    
    await waitFor(() => {
      expect(screen.getByText('Dialog Title')).toBeInTheDocument();
    });
  });

  it('should close on close button', async () => {
    const { Dialog, DialogTrigger, DialogContent, DialogTitle, DialogClose } = await import('./dialog');
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
          <DialogClose data-testid="dialog-close">Close Dialog</DialogClose>
        </DialogContent>
      </Dialog>
    );
    
    fireEvent.click(screen.getByText('Open'));
    await waitFor(() => expect(screen.getByText('Title')).toBeInTheDocument());
    
    // Use role to find the close button
    fireEvent.click(screen.getByTestId('dialog-close'));
    await waitFor(() => expect(screen.queryByText('Title')).not.toBeInTheDocument());
  });
});

// ============================================
// SELECT COMPONENT TESTS  
// Note: Radix UI Select has scrollIntoView issues in jsdom
// Testing trigger rendering and aria attributes
// ============================================

describe('Select', () => {
  it('should render select trigger', async () => {
    const { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } = await import('./select');
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Select an option" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
        </SelectContent>
      </Select>
    );
    
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('should have correct placeholder', async () => {
    const { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } = await import('./select');
    render(
      <Select>
        <SelectTrigger>
          <SelectValue placeholder="Choose..." />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="a">A</SelectItem>
        </SelectContent>
      </Select>
    );
    
    expect(screen.getByText('Choose...')).toBeInTheDocument();
  });
});

// ============================================
// TABS COMPONENT TESTS
// ============================================

describe('Tabs', () => {
  it('should render tabs', async () => {
    const { Tabs, TabsList, TabsTrigger, TabsContent } = await import('./tabs');
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    );
    
    expect(screen.getByRole('tablist')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /tab 1/i })).toBeInTheDocument();
  });

  it('should have correct number of tabs', async () => {
    const { Tabs, TabsList, TabsTrigger, TabsContent } = await import('./tabs');
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          <TabsTrigger value="tab3">Tab 3</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
        <TabsContent value="tab3">Content 3</TabsContent>
      </Tabs>
    );
    
    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(3);
  });
});

// ============================================
// DROPDOWN MENU TESTS
// Note: Radix UI dropdown requires special DOM mocking 
// Focus on trigger rendering to avoid scrollIntoView issues
// ============================================

describe('DropdownMenu', () => {
  it('should render dropdown trigger', async () => {
    const { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } = await import('./dropdown-menu');
    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Open Menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Item 1</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
    
    expect(screen.getByText('Open Menu')).toBeInTheDocument();
  });

  it('should have correct aria attributes', async () => {
    const { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } = await import('./dropdown-menu');
    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Open</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Item 1</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
    
    const trigger = screen.getByText('Open');
    expect(trigger).toHaveAttribute('aria-haspopup', 'menu');
  });
});

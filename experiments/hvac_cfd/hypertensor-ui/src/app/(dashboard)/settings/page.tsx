/**
 * Settings Page
 * 
 * Application settings and preferences.
 */

'use client';

import { useState } from 'react';
import { Settings, Monitor, Palette, Bell, Shield, HardDrive } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Sidebar,
  Header,
  DashboardShell,
  PageHeader,
  SectionCard,
} from '@/components/layout';
import { useToast } from '@/components/ui/use-toast';
import { useLocalStorage } from '@/hooks';

// ============================================
// SETTINGS SECTION
// ============================================

interface SettingItemProps {
  icon: React.ElementType;
  label: string;
  description: string;
  children: React.ReactNode;
}

function SettingItem({ icon: Icon, label, description, children }: SettingItemProps) {
  return (
    <div className="flex items-center justify-between py-4">
      <div className="flex items-center gap-4">
        <div className="p-2 rounded-lg bg-muted">
          <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
        <div>
          <Label className="text-sm font-medium">{label}</Label>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </div>
      <div>{children}</div>
    </div>
  );
}

// ============================================
// MAIN PAGE
// ============================================

export default function SettingsPage() {
  const { toast } = useToast();
  const [autoRefresh, setAutoRefresh] = useLocalStorage('settings-autoRefresh', true);
  const [notifications, setNotifications] = useLocalStorage('settings-notifications', true);
  const [gpuAcceleration, setGpuAcceleration] = useLocalStorage('settings-gpu', true);
  const [precision, setPrecision] = useLocalStorage('settings-precision', 'double');

  const handleSave = () => {
    toast({
      title: 'Settings saved',
      description: 'Your preferences have been updated.',
    });
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <DashboardShell>
          <div className="space-y-6 max-w-3xl">
            <PageHeader
              title="Settings"
              description="Configure application preferences and defaults"
              breadcrumbs={[
                { label: 'Dashboard', href: '/' },
                { label: 'Settings' },
              ]}
            />

            {/* Display Settings */}
            <SectionCard title="Display" description="Interface and visualization preferences">
              <div className="divide-y">
                <SettingItem
                  icon={Monitor}
                  label="Auto-refresh"
                  description="Automatically update simulation status"
                >
                  <Switch
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                  />
                </SettingItem>

                <SettingItem
                  icon={Palette}
                  label="Theme"
                  description="Color scheme preference"
                >
                  <Select defaultValue="system">
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </SettingItem>
              </div>
            </SectionCard>

            {/* Notifications */}
            <SectionCard title="Notifications" description="Alert and notification settings">
              <div className="divide-y">
                <SettingItem
                  icon={Bell}
                  label="Desktop notifications"
                  description="Show alerts when simulations complete"
                >
                  <Switch
                    checked={notifications}
                    onCheckedChange={setNotifications}
                  />
                </SettingItem>
              </div>
            </SectionCard>

            {/* Solver Defaults */}
            <SectionCard title="Solver Defaults" description="Default simulation settings">
              <div className="divide-y">
                <SettingItem
                  icon={HardDrive}
                  label="GPU Acceleration"
                  description="Use GPU for tensor operations by default"
                >
                  <Switch
                    checked={gpuAcceleration}
                    onCheckedChange={setGpuAcceleration}
                  />
                </SettingItem>

                <SettingItem
                  icon={Shield}
                  label="Numerical Precision"
                  description="Default floating-point precision"
                >
                  <Select value={precision} onValueChange={setPrecision}>
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="float32">Float32</SelectItem>
                      <SelectItem value="double">Double</SelectItem>
                    </SelectContent>
                  </Select>
                </SettingItem>
              </div>
            </SectionCard>

            {/* Save Button */}
            <div className="flex justify-end">
              <Button onClick={handleSave}>
                Save Settings
              </Button>
            </div>
          </div>
        </DashboardShell>
      </div>
    </div>
  );
}

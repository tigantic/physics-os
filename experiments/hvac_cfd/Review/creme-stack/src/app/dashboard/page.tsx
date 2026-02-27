import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui';
import { TrendingUp, Users, DollarSign, Activity } from 'lucide-react';

export const metadata = {
  title: 'Dashboard',
};

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome back! Here&apos;s what&apos;s happening.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Revenue"
          value="$45,231.89"
          change="+20.1%"
          changeType="positive"
          icon={<DollarSign className="h-4 w-4" />}
        />
        <StatsCard
          title="Subscriptions"
          value="+2,350"
          change="+180.1%"
          changeType="positive"
          icon={<Users className="h-4 w-4" />}
        />
        <StatsCard
          title="Sales"
          value="+12,234"
          change="+19%"
          changeType="positive"
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <StatsCard
          title="Active Now"
          value="+573"
          change="+201"
          changeType="positive"
          icon={<Activity className="h-4 w-4" />}
        />
      </div>

      {/* Content Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        {/* Overview Card */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <CardTitle>Overview</CardTitle>
            <CardDescription>
              Your performance metrics for this period.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed">
              <p className="text-muted-foreground">
                Chart component goes here
                <br />
                <span className="text-sm">(use Recharts or Chart.js)</span>
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity Card */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>
              Latest updates from your team.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex items-center gap-4">
                  <div className="h-10 w-10 rounded-full bg-muted" />
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium">User action {i}</p>
                    <p className="text-xs text-muted-foreground">
                      {i} hour{i > 1 ? 's' : ''} ago
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ============================================
// STATS CARD COMPONENT
// ============================================

function StatsCard({
  title,
  value,
  change,
  changeType,
  icon,
}: {
  title: string;
  value: string;
  change: string;
  changeType: 'positive' | 'negative' | 'neutral';
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p
          className={`text-xs ${
            changeType === 'positive'
              ? 'text-green-600 dark:text-green-400'
              : changeType === 'negative'
                ? 'text-red-600 dark:text-red-400'
                : 'text-muted-foreground'
          }`}
        >
          {change} from last month
        </p>
      </CardContent>
    </Card>
  );
}

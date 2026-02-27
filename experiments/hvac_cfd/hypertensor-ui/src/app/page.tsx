/**
 * HyperTensor Root Page
 * 
 * Redirects to the CFD dashboard. This is a production application,
 * not a template showcase.
 */

import { redirect } from 'next/navigation';

export default function HomePage() {
  redirect('/simulations');
}

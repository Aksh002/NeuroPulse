import './globals.css';
export const metadata = { title: 'NeuroPulse · Degradation Observatory', description: 'Neural predictive-maintenance research and historical replay.' };
export default function RootLayout({children}: {children: React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}

import Link from "next/link";
import BottomToBullChecklist from "@/app/components/BottomToBullChecklist";

export default function BottomToBullPage() {
  return (
    <div className="branden-journal-content">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Market recovery</p>
            <h1>Bottom to Bull</h1>
            <span>Track the technical conditions that confirm a durable market recovery.</span>
          </div>
          <nav className="branden-route-nav" aria-label="Branden journal pages">
            <Link href="/journal/branden/dashboard">Dashboard</Link>
            <Link className="active" href="/journal/branden/bottom-to-bull">Bottom to Bull</Link>
            <Link href="/journal/branden/market-cycle">Market Cycle</Link>
          </nav>
        </header>

        <BottomToBullChecklist />
      </div>
  );
}

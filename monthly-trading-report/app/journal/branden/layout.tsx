import BrandenJournalLayoutClient from "./BrandenJournalLayoutClient";

export default function BrandenJournalLayout({ children }: { children: React.ReactNode }) {
  return <BrandenJournalLayoutClient>{children}</BrandenJournalLayoutClient>;
}
